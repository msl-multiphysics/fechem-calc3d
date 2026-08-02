"""Tet4 / Hex8 / Tri3 / Quad4 Gauss rules and Dom / Bnd integral helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_G = 1.0 / np.sqrt(3.0)  # 0.5773502691896257

# Gmsh Tet4 face table (outward normals for positive-volume tets).
TET4_FACES: list[list[int]] = [
    [0, 2, 1],
    [0, 1, 3],
    [0, 3, 2],
    [1, 2, 3],
]

# Gmsh Hex8 face table (outward normals for positive-volume hexes).
HEX8_FACES: list[list[int]] = [
    [0, 3, 2, 1],
    [0, 1, 5, 4],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [4, 5, 6, 7],
]

# Reference-node coordinates matching shape-function definitions.
TET4_REF = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)
HEX8_REF = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Shape functions / rules
# ---------------------------------------------------------------------------

def tet4_quad_rule() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (w, a, b, c) for the 4-point Tet4 Hammer rule."""
    alpha = 0.58541020
    beta = 0.13819660
    w = np.full(4, 1.0 / 24.0, dtype=float)
    a = np.array([beta, alpha, beta, beta], dtype=float)
    b = np.array([beta, beta, alpha, beta], dtype=float)
    c = np.array([beta, beta, beta, alpha], dtype=float)
    return w, a, b, c


def hex8_quad_rule() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (w, a, b, c) for the 2x2x2 Hex8 Gauss rule."""
    w = np.ones(8, dtype=float)
    a = np.array([-_G, _G, -_G, _G, -_G, _G, -_G, _G], dtype=float)
    b = np.array([-_G, -_G, _G, _G, -_G, -_G, _G, _G], dtype=float)
    c = np.array([-_G, -_G, -_G, -_G, _G, _G, _G, _G], dtype=float)
    return w, a, b, c


def tri3_quad_rule() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (w, a, b) for the 3-point Tri3 rule."""
    w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=float)
    a = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=float)
    b = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], dtype=float)
    return w, a, b


def quad4_quad_rule() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (w, a, b) for the 2x2 Quad4 Gauss rule."""
    w = np.ones(4, dtype=float)
    a = np.array([-_G, -_G, _G, _G], dtype=float)
    b = np.array([-_G, _G, -_G, _G], dtype=float)
    return w, a, b


def tet4_eval(a: float, b: float, c: float) -> np.ndarray:
    return np.array([1.0 - a - b - c, a, b, c], dtype=float)


def tet4_grad_ref(a: float, b: float, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dn_da = np.array([-1.0, 1.0, 0.0, 0.0], dtype=float)
    dn_db = np.array([-1.0, 0.0, 1.0, 0.0], dtype=float)
    dn_dc = np.array([-1.0, 0.0, 0.0, 1.0], dtype=float)
    return dn_da, dn_db, dn_dc


def hex8_eval(a: float, b: float, c: float) -> np.ndarray:
    return np.array(
        [
            0.125 * (1.0 - a) * (1.0 - b) * (1.0 - c),
            0.125 * (1.0 + a) * (1.0 - b) * (1.0 - c),
            0.125 * (1.0 + a) * (1.0 + b) * (1.0 - c),
            0.125 * (1.0 - a) * (1.0 + b) * (1.0 - c),
            0.125 * (1.0 - a) * (1.0 - b) * (1.0 + c),
            0.125 * (1.0 + a) * (1.0 - b) * (1.0 + c),
            0.125 * (1.0 + a) * (1.0 + b) * (1.0 + c),
            0.125 * (1.0 - a) * (1.0 + b) * (1.0 + c),
        ],
        dtype=float,
    )


def hex8_grad_ref(a: float, b: float, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dn_da = np.array(
        [
            -0.125 * (1.0 - b) * (1.0 - c),
            0.125 * (1.0 - b) * (1.0 - c),
            0.125 * (1.0 + b) * (1.0 - c),
            -0.125 * (1.0 + b) * (1.0 - c),
            -0.125 * (1.0 - b) * (1.0 + c),
            0.125 * (1.0 - b) * (1.0 + c),
            0.125 * (1.0 + b) * (1.0 + c),
            -0.125 * (1.0 + b) * (1.0 + c),
        ],
        dtype=float,
    )
    dn_db = np.array(
        [
            -0.125 * (1.0 - a) * (1.0 - c),
            -0.125 * (1.0 + a) * (1.0 - c),
            0.125 * (1.0 + a) * (1.0 - c),
            0.125 * (1.0 - a) * (1.0 - c),
            -0.125 * (1.0 - a) * (1.0 + c),
            -0.125 * (1.0 + a) * (1.0 + c),
            0.125 * (1.0 + a) * (1.0 + c),
            0.125 * (1.0 - a) * (1.0 + c),
        ],
        dtype=float,
    )
    dn_dc = np.array(
        [
            -0.125 * (1.0 - a) * (1.0 - b),
            -0.125 * (1.0 + a) * (1.0 - b),
            -0.125 * (1.0 + a) * (1.0 + b),
            -0.125 * (1.0 - a) * (1.0 + b),
            0.125 * (1.0 - a) * (1.0 - b),
            0.125 * (1.0 + a) * (1.0 - b),
            0.125 * (1.0 + a) * (1.0 + b),
            0.125 * (1.0 - a) * (1.0 + b),
        ],
        dtype=float,
    )
    return dn_da, dn_db, dn_dc


def tri3_eval(a: float, b: float) -> np.ndarray:
    return np.array([1.0 - a - b, a, b], dtype=float)


def tri3_grad_ref(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    dn_da = np.array([-1.0, 1.0, 0.0], dtype=float)
    dn_db = np.array([-1.0, 0.0, 1.0], dtype=float)
    return dn_da, dn_db


def quad4_eval(a: float, b: float) -> np.ndarray:
    return np.array(
        [
            0.25 * (1.0 - a) * (1.0 - b),
            0.25 * (1.0 + a) * (1.0 - b),
            0.25 * (1.0 + a) * (1.0 + b),
            0.25 * (1.0 - a) * (1.0 + b),
        ],
        dtype=float,
    )


def quad4_grad_ref(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    dn_da = np.array(
        [
            -0.25 * (1.0 - b),
            0.25 * (1.0 - b),
            0.25 * (1.0 + b),
            -0.25 * (1.0 + b),
        ],
        dtype=float,
    )
    dn_db = np.array(
        [
            -0.25 * (1.0 - a),
            -0.25 * (1.0 + a),
            0.25 * (1.0 + a),
            0.25 * (1.0 - a),
        ],
        dtype=float,
    )
    return dn_da, dn_db


def _is_tetra(cell_type: str) -> bool:
    return cell_type.startswith("tetra")


def _is_hex(cell_type: str) -> bool:
    return cell_type.startswith("hexahedron") or cell_type == "hex"


def _is_triangle(cell_type: str) -> bool:
    return cell_type.startswith("triangle")


def _is_quad(cell_type: str) -> bool:
    return cell_type.startswith("quad")


def volume_faces(cell_type: str) -> list[list[int]]:
    if _is_tetra(cell_type):
        return TET4_FACES
    if _is_hex(cell_type):
        return HEX8_FACES
    raise ValueError(f"Unsupported volume cell type {cell_type!r}")


def physical_grad(
    node_xyz: np.ndarray,
    dn_da: np.ndarray,
    dn_db: np.ndarray,
    dn_dc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Map reference gradients to physical (gnx, gny, gnz) and return det(J)."""
    # J columns: ∂x/∂a, ∂x/∂b, ∂x/∂c
    j = np.zeros((3, 3), dtype=float)
    for k, dn in enumerate((dn_da, dn_db, dn_dc)):
        j[0, k] = float(np.dot(dn, node_xyz[:, 0]))
        j[1, k] = float(np.dot(dn, node_xyz[:, 1]))
        j[2, k] = float(np.dot(dn, node_xyz[:, 2]))
    det = float(np.linalg.det(j))
    n = len(dn_da)
    if abs(det) < 1e-30:
        z = np.zeros(n, dtype=float)
        return z, z, z, det
    inv = np.linalg.inv(j)
    # physical gradient: [dN/dx; dN/dy; dN/dz] = inv.T @ [dN/da; dN/db; dN/dc]
    # gnx_i = inv[0,0]*dna_i + inv[1,0]*dnb_i + inv[2,0]*dnc_i
    gnx = inv[0, 0] * dn_da + inv[1, 0] * dn_db + inv[2, 0] * dn_dc
    gny = inv[0, 1] * dn_da + inv[1, 1] * dn_db + inv[2, 1] * dn_dc
    gnz = inv[0, 2] * dn_da + inv[1, 2] * dn_db + inv[2, 2] * dn_dc
    return gnx, gny, gnz, det


def shape_at_ref_3d(
    cell_type: str, xi: float, eta: float, zeta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (N, dN/dξ, dN/dη, dN/dζ) at volume reference coords."""
    if _is_tetra(cell_type):
        return tet4_eval(xi, eta, zeta), *tet4_grad_ref(xi, eta, zeta)
    if _is_hex(cell_type):
        return hex8_eval(xi, eta, zeta), *hex8_grad_ref(xi, eta, zeta)
    raise ValueError(f"Unsupported volume cell type {cell_type!r}")


def face_shape_at_ref(
    face_type: str, a: float, b: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (N, dN/da, dN/db) on a Tri3 or Quad4 face."""
    if _is_triangle(face_type):
        return tri3_eval(a, b), *tri3_grad_ref(a, b)
    if _is_quad(face_type):
        return quad4_eval(a, b), *quad4_grad_ref(a, b)
    raise ValueError(f"Unsupported face cell type {face_type!r}")


def face_type_from_nnodes(n: int) -> str:
    if n == 3:
        return "triangle"
    if n == 4:
        return "quad"
    raise ValueError(f"Unsupported face node count {n}")


# ---------------------------------------------------------------------------
# Precomputed Dom tables
# ---------------------------------------------------------------------------

@dataclass
class DomQuadTables:
    """Per-element domain quadrature data (Dom-local indexing)."""

    cell_type: str
    num_quad: list[int] = field(default_factory=list)
    quad_w: list[np.ndarray] = field(default_factory=list)
    quad_n: list[np.ndarray] = field(default_factory=list)  # [e](q, v)
    quad_gnx: list[np.ndarray] = field(default_factory=list)
    quad_gny: list[np.ndarray] = field(default_factory=list)
    quad_gnz: list[np.ndarray] = field(default_factory=list)
    jac_det: list[np.ndarray] = field(default_factory=list)
    quad_xyz: list[np.ndarray] = field(default_factory=list)  # [e](q, 3)


def build_dom_quad_tables(
    points_xyz: np.ndarray,
    cells_local: np.ndarray,
    cell_type: str,
) -> DomQuadTables:
    """Precompute Dom quadrature tables for a uniform Tet4 or Hex8 domain."""
    tables = DomQuadTables(cell_type=cell_type)
    if _is_tetra(cell_type):
        w_ref, a_ref, b_ref, c_ref = tet4_quad_rule()
    elif _is_hex(cell_type):
        w_ref, a_ref, b_ref, c_ref = hex8_quad_rule()
    else:
        raise ValueError(f"Unsupported domain cell type {cell_type!r}")

    nq = len(w_ref)
    for cell in cells_local:
        node_xyz = points_xyz[cell]
        qw = np.empty(nq, dtype=float)
        qn = np.empty((nq, len(cell)), dtype=float)
        gnx = np.empty((nq, len(cell)), dtype=float)
        gny = np.empty((nq, len(cell)), dtype=float)
        gnz = np.empty((nq, len(cell)), dtype=float)
        jdet = np.empty(nq, dtype=float)
        qxyz = np.empty((nq, 3), dtype=float)
        for q in range(nq):
            n, dna, dnb, dnc = shape_at_ref_3d(
                cell_type, float(a_ref[q]), float(b_ref[q]), float(c_ref[q])
            )
            gx, gy, gz, det = physical_grad(node_xyz, dna, dnb, dnc)
            qw[q] = w_ref[q]
            qn[q] = n
            gnx[q] = gx
            gny[q] = gy
            gnz[q] = gz
            jdet[q] = det
            qxyz[q] = n @ node_xyz
        tables.num_quad.append(nq)
        tables.quad_w.append(qw)
        tables.quad_n.append(qn)
        tables.quad_gnx.append(gnx)
        tables.quad_gny.append(gny)
        tables.quad_gnz.append(gnz)
        tables.jac_det.append(jdet)
        tables.quad_xyz.append(qxyz)
    return tables


# ---------------------------------------------------------------------------
# Face ↔ volume parent mapping
# ---------------------------------------------------------------------------

def face_directed_index(cell: np.ndarray, face_nodes: np.ndarray) -> int:
    """Return face index on a volume cell whose local nodes match ``face_nodes``."""
    key = frozenset(int(x) for x in face_nodes)
    faces = volume_faces("tetra" if len(cell) == 4 else "hexahedron")
    for i, face in enumerate(faces):
        ordered = [int(cell[j]) for j in face]
        if frozenset(ordered) == key:
            return i
    raise ValueError(f"Face {face_nodes} not found on cell {cell}")


def parent_ref_on_face(
    cell_type: str,
    face: int,
    face_a: float,
    face_b: float,
) -> tuple[float, float, float]:
    """Map Tri3/Quad4 face params to volume reference (ξ, η, ζ)."""
    faces = volume_faces(cell_type)
    local_idxs = faces[face]
    if _is_tetra(cell_type):
        n_face = tri3_eval(face_a, face_b)
        ref = TET4_REF[local_idxs]
    elif _is_hex(cell_type):
        n_face = quad4_eval(face_a, face_b)
        ref = HEX8_REF[local_idxs]
    else:
        raise ValueError(f"Unsupported cell type {cell_type!r}")
    xyz = n_face @ ref
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def face_unit_n_and_jac(
    face_xyz: np.ndarray,
    face_a: float,
    face_b: float,
    cell_centroid: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Unit outward normal, surface jacobian ||ta×tb||, and physical qpt."""
    n_face = len(face_xyz)
    ftype = face_type_from_nnodes(n_face)
    n, dna, dnb = face_shape_at_ref(ftype, face_a, face_b)
    ta = dna @ face_xyz
    tb = dnb @ face_xyz
    n_out = np.cross(ta, tb)
    mid = n @ face_xyz
    if np.dot(n_out, mid - cell_centroid) < 0.0:
        n_out = -n_out
    jac = float(np.linalg.norm(n_out))
    if jac < 1e-30:
        return np.zeros(3, dtype=float), 0.0, mid
    return n_out / jac, jac, mid


def eval_dom_at_face_qpt(
    cell: np.ndarray,
    cell_type: str,
    points_xyz: np.ndarray,
    face_local_nodes: np.ndarray,
    face_a: float,
    face_b: float,
    scl_nodal: np.ndarray | None = None,
    vec_nodal: np.ndarray | None = None,
) -> dict:
    """Evaluate Dom fields / gradients at a Tri3/Quad4 face quadrature point.

    Returns dict with keys: u, grad (3,), vel (3,), grad_v (3,3) as available.
    ``grad_v[i, j]`` = ∂v_i / ∂x_j.
    """
    face = face_directed_index(cell, face_local_nodes)
    xi, eta, zeta = parent_ref_on_face(cell_type, face, face_a, face_b)
    n, dna, dnb, dnc = shape_at_ref_3d(cell_type, xi, eta, zeta)
    node_xyz = points_xyz[cell]
    gnx, gny, gnz, _ = physical_grad(node_xyz, dna, dnb, dnc)

    out: dict = {}
    if scl_nodal is not None:
        vals = scl_nodal[cell]
        out["u"] = float(n @ vals)
        out["grad"] = np.array(
            [float(gnx @ vals), float(gny @ vals), float(gnz @ vals)],
            dtype=float,
        )
    if vec_nodal is not None:
        vx = vec_nodal[cell, 0]
        vy = vec_nodal[cell, 1]
        vz = vec_nodal[cell, 2]
        out["vel"] = np.array(
            [float(n @ vx), float(n @ vy), float(n @ vz)], dtype=float
        )
        out["grad_v"] = np.array(
            [
                [float(gnx @ vx), float(gny @ vx), float(gnz @ vx)],
                [float(gnx @ vy), float(gny @ vy), float(gnz @ vy)],
                [float(gnx @ vz), float(gny @ vz), float(gnz @ vz)],
            ],
            dtype=float,
        )
    return out
