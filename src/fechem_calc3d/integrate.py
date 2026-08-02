"""Gauss quadrature volume and surface integrals."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fields import eval_prop_scl, eval_prop_scl_of_vec, eval_prop_vec
from .quadrature import (
    DomQuadTables,
    eval_dom_at_face_qpt,
    face_type_from_nnodes,
    face_unit_n_and_jac,
    quad4_quad_rule,
    tri3_quad_rule,
    volume_faces,
)


def build_face_adjacency(
    dom_cells: dict[int, np.ndarray],
    dom_cell_types: dict[int, str],
) -> dict[frozenset[int], list[tuple[int, np.ndarray, np.ndarray]]]:
    """
    Map unordered face node frozenset (Dom-local ids) → list of
    (dom_id, cell_node_ids, face_ordered_outward).
    """
    adj: dict[frozenset[int], list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for dom_id, cells in dom_cells.items():
        cell_type = dom_cell_types[dom_id]
        faces = volume_faces(cell_type)
        for cell in cells:
            for face in faces:
                ordered = np.array([int(cell[i]) for i in face], dtype=int)
                key = frozenset(int(x) for x in ordered)
                adj.setdefault(key, []).append((dom_id, cell.copy(), ordered))
    return adj


def _resolve_face_cell(
    face: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    face_adj: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ordered_local_face_nodes, parent_cell)."""
    local_nodes = []
    for gid in face:
        gid = int(gid)
        if gid not in global_to_local:
            raise ValueError(
                f"Boundary face node {gid} is not in domain {field_dom}"
            )
        local_nodes.append(global_to_local[gid])
    key = frozenset(local_nodes)
    candidates = [item for item in face_adj.get(key, []) if item[0] == field_dom]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one adjacent cell in domain {field_dom} for face "
            f"{list(map(int, face))}, found {len(candidates)}"
        )
    _, cell, ordered = candidates[0]
    return ordered, cell


def _iter_face_qpts(
    points: np.ndarray,
    face: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xyz: np.ndarray,
    face_adj: dict,
):
    """Yield (ordered_local, cell, face_a, face_b, unit_n, w*jac) for each face qpt."""
    del points  # global coords unused; Dom-local tables carry XYZ
    ordered, cell = _resolve_face_cell(face, global_to_local, field_dom, face_adj)
    face_xyz = dom_points_xyz[ordered]
    centroid = np.mean(dom_points_xyz[cell], axis=0)
    n_face = len(ordered)
    ftype = face_type_from_nnodes(n_face)
    if ftype.startswith("triangle"):
        w_ref, a_ref, b_ref = tri3_quad_rule()
    else:
        w_ref, a_ref, b_ref = quad4_quad_rule()
    for a, b, w in zip(a_ref, b_ref, w_ref):
        a = float(a)
        b = float(b)
        unit_n, jac, _ = face_unit_n_and_jac(face_xyz, a, b, centroid)
        if jac < 1e-30:
            continue
        yield ordered, cell, a, b, unit_n, jac * float(w)


def volint_scl_src(
    tables: DomQuadTables,
    cells_local: np.ndarray,
    src: Any,
    unk: np.ndarray,
) -> float:
    """∫ src(u) dΩ over Dom cells."""
    total = 0.0
    for eid, cell in enumerate(cells_local):
        nq = tables.num_quad[eid]
        qw = tables.quad_w[eid]
        qn = tables.quad_n[eid]
        jdet = tables.jac_det[eid]
        u_nodes = unk[cell]
        for q in range(nq):
            u_q = float(qn[q] @ u_nodes)
            s = eval_prop_scl(src, u_q)
            total += s * float(qw[q]) * abs(float(jdet[q]))
    return total


def volint_vec_src(
    tables: DomQuadTables,
    cells_local: np.ndarray,
    src: Any,
    fce: np.ndarray,
) -> np.ndarray:
    """∫ src(f) dΩ over Dom cells; returns length-3 vector."""
    total = np.zeros(3, dtype=float)
    for eid, cell in enumerate(cells_local):
        nq = tables.num_quad[eid]
        qw = tables.quad_w[eid]
        qn = tables.quad_n[eid]
        jdet = tables.jac_det[eid]
        f_nodes = fce[cell]
        for q in range(nq):
            f_q = qn[q] @ f_nodes
            s = eval_prop_vec(src, f_q)
            total += s * float(qw[q]) * abs(float(jdet[q]))
    return total


def surfint_scl_diff(
    points: np.ndarray,
    face_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xyz: np.ndarray,
    cell_type: str,
    face_adj: dict,
    diff: Any,
    unk: np.ndarray,
) -> float:
    """∫ -diff(u) (∇u · n) dS on Bnd/Itf faces."""
    total = 0.0
    for face in face_cells:
        for ordered, cell, a, b, unit_n, wjac in _iter_face_qpts(
            points, face, global_to_local, field_dom, dom_points_xyz, face_adj
        ):
            ev = eval_dom_at_face_qpt(
                cell,
                cell_type,
                dom_points_xyz,
                ordered,
                a,
                b,
                scl_nodal=unk,
            )
            d = eval_prop_scl(diff, ev["u"])
            total += -d * float(np.dot(ev["grad"], unit_n)) * wjac
    return total


def surfint_scl_adv(
    points: np.ndarray,
    face_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xyz: np.ndarray,
    cell_type: str,
    face_adj: dict,
    wgt: Any,
    vel: np.ndarray,
    unk: np.ndarray,
) -> float:
    """∫ wgt(u) u (v · n) dS on Bnd/Itf faces."""
    total = 0.0
    for face in face_cells:
        for ordered, cell, a, b, unit_n, wjac in _iter_face_qpts(
            points, face, global_to_local, field_dom, dom_points_xyz, face_adj
        ):
            ev = eval_dom_at_face_qpt(
                cell,
                cell_type,
                dom_points_xyz,
                ordered,
                a,
                b,
                scl_nodal=unk,
                vec_nodal=vel,
            )
            w = eval_prop_scl(wgt, ev["u"])
            vn = float(np.dot(ev["vel"], unit_n))
            total += w * ev["u"] * vn * wjac
    return total


def surfint_vec_visc(
    points: np.ndarray,
    face_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xyz: np.ndarray,
    cell_type: str,
    face_adj: dict,
    visc: Any,
    vel: np.ndarray,
) -> np.ndarray:
    """∫ -(τ · n) dS, the viscous force of fluid on the boundary."""
    total = np.zeros(3, dtype=float)
    for face in face_cells:
        for ordered, cell, a, b, unit_n, wjac in _iter_face_qpts(
            points, face, global_to_local, field_dom, dom_points_xyz, face_adj
        ):
            ev = eval_dom_at_face_qpt(
                cell,
                cell_type,
                dom_points_xyz,
                ordered,
                a,
                b,
                vec_nodal=vel,
            )
            mu = eval_prop_scl_of_vec(visc, ev["vel"])
            gv = ev["grad_v"]
            div = float(gv[0, 0] + gv[1, 1] + gv[2, 2])
            tau = np.zeros((3, 3), dtype=float)
            for i in range(3):
                for j in range(3):
                    tau[i, j] = mu * (gv[i, j] + gv[j, i])
                tau[i, i] -= (2.0 / 3.0) * mu * div
            total += -(tau @ unit_n) * wjac
    return total


def surfint_vec_adv(
    points: np.ndarray,
    face_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xyz: np.ndarray,
    cell_type: str,
    face_adj: dict,
    den: Any,
    vel: np.ndarray,
) -> np.ndarray:
    """∫ ρ (v · n) v dS on Bnd/Itf faces."""
    total = np.zeros(3, dtype=float)
    for face in face_cells:
        for ordered, cell, a, b, unit_n, wjac in _iter_face_qpts(
            points, face, global_to_local, field_dom, dom_points_xyz, face_adj
        ):
            ev = eval_dom_at_face_qpt(
                cell,
                cell_type,
                dom_points_xyz,
                ordered,
                a,
                b,
                vec_nodal=vel,
            )
            rho = eval_prop_scl_of_vec(den, ev["vel"])
            vn = float(np.dot(ev["vel"], unit_n))
            total += rho * vn * ev["vel"] * wjac
    return total


def surfint_vec_pres(
    points: np.ndarray,
    face_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xyz: np.ndarray,
    cell_type: str,
    face_adj: dict,
    pres: np.ndarray,
) -> np.ndarray:
    """∫ p n dS on Bnd/Itf faces."""
    total = np.zeros(3, dtype=float)
    for face in face_cells:
        for ordered, cell, a, b, unit_n, wjac in _iter_face_qpts(
            points, face, global_to_local, field_dom, dom_points_xyz, face_adj
        ):
            ev = eval_dom_at_face_qpt(
                cell,
                cell_type,
                dom_points_xyz,
                ordered,
                a,
                b,
                scl_nodal=pres,
            )
            total += ev["u"] * unit_n * wjac
    return total
