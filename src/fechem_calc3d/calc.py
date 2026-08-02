"""Calc3D: Gmsh / structured mesh + FEChem VTU quadrature integrals."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fields import read_vtu_values, remap_vtu_to_domain
from .integrate import (
    build_face_adjacency,
    surfint_scl_adv,
    surfint_scl_diff,
    surfint_vec_adv,
    surfint_vec_pres,
    surfint_vec_visc,
    volint_scl_src,
    volint_vec_src,
)
from .quadrature import build_dom_quad_tables
from .viz import MeshSnapshot, RegionMesh


def _snapshot_from_bounds(
    x_min: float,
    y_min: float,
    z_min: float,
    x_max: float,
    y_max: float,
    z_max: float,
    num_elem_x: int,
    num_elem_y: int,
    num_elem_z: int,
) -> MeshSnapshot:
    """Structured Hex8 mesh matching Rust ``Mesh::new_from_bounds``."""
    if x_min >= x_max:
        raise ValueError(f"Invalid x bounds: {x_min} >= {x_max}")
    if y_min >= y_max:
        raise ValueError(f"Invalid y bounds: {y_min} >= {y_max}")
    if z_min >= z_max:
        raise ValueError(f"Invalid z bounds: {z_min} >= {z_max}")
    if num_elem_x < 1 or num_elem_y < 1 or num_elem_z < 1:
        raise ValueError("num_elem_x/y/z must be >= 1")

    dx = (x_max - x_min) / float(num_elem_x)
    dy = (y_max - y_min) / float(num_elem_y)
    dz = (z_max - z_min) / float(num_elem_z)
    nx = num_elem_x + 1
    ny = num_elem_y + 1
    nz = num_elem_z + 1
    stride_i = 1
    stride_j = nx
    stride_k = nx * ny

    def nid(i: int, j: int, k: int) -> int:
        return i * stride_i + j * stride_j + k * stride_k

    points = np.array(
        [
            [x_min + i * dx, y_min + j * dy, z_min + k * dz]
            for k in range(nz)
            for j in range(ny)
            for i in range(nx)
        ],
        dtype=float,
    )

    hexes = []
    for k in range(num_elem_z):
        for j in range(num_elem_y):
            for i in range(num_elem_x):
                n0 = nid(i, j, k)
                n1 = nid(i + 1, j, k)
                n2 = nid(i + 1, j + 1, k)
                n3 = nid(i, j + 1, k)
                n4 = nid(i, j, k + 1)
                n5 = nid(i + 1, j, k + 1)
                n6 = nid(i + 1, j + 1, k + 1)
                n7 = nid(i, j + 1, k + 1)
                hexes.append([n0, n1, n2, n3, n4, n5, n6, n7])
    hexes_arr = np.asarray(hexes, dtype=int)

    # Boundaries: 0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+ (outward RH order)
    xm, xp, ym, yp, zm, zp = [], [], [], [], [], []
    for k in range(num_elem_z):
        for j in range(num_elem_y):
            xm.append(
                [
                    nid(0, j, k),
                    nid(0, j, k + 1),
                    nid(0, j + 1, k + 1),
                    nid(0, j + 1, k),
                ]
            )
            xp.append(
                [
                    nid(num_elem_x, j, k),
                    nid(num_elem_x, j + 1, k),
                    nid(num_elem_x, j + 1, k + 1),
                    nid(num_elem_x, j, k + 1),
                ]
            )
    for k in range(num_elem_z):
        for i in range(num_elem_x):
            ym.append(
                [
                    nid(i, 0, k),
                    nid(i + 1, 0, k),
                    nid(i + 1, 0, k + 1),
                    nid(i, 0, k + 1),
                ]
            )
            yp.append(
                [
                    nid(i, num_elem_y, k),
                    nid(i, num_elem_y, k + 1),
                    nid(i + 1, num_elem_y, k + 1),
                    nid(i + 1, num_elem_y, k),
                ]
            )
    for j in range(num_elem_y):
        for i in range(num_elem_x):
            zm.append(
                [
                    nid(i, j, 0),
                    nid(i, j + 1, 0),
                    nid(i + 1, j + 1, 0),
                    nid(i + 1, j, 0),
                ]
            )
            zp.append(
                [
                    nid(i, j, num_elem_z),
                    nid(i + 1, j, num_elem_z),
                    nid(i + 1, j + 1, num_elem_z),
                    nid(i, j + 1, num_elem_z),
                ]
            )

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)

    # Gmsh-style tags start at 1 → FEChem indices = tag - 1
    dom = RegionMesh(
        tag=1,
        name="Domain_0",
        dim=3,
        cells=hexes_arr,
        cell_type="hexahedron",
        label_points=np.array([[cx, cy, cz]]),
    )
    boundaries = [
        RegionMesh(
            tag=1,
            name="Boundary_0",
            dim=2,
            cells=np.asarray(xm, dtype=int),
            cell_type="quad",
            label_points=np.array([[x_min, cy, cz]]),
        ),
        RegionMesh(
            tag=2,
            name="Boundary_1",
            dim=2,
            cells=np.asarray(xp, dtype=int),
            cell_type="quad",
            label_points=np.array([[x_max, cy, cz]]),
        ),
        RegionMesh(
            tag=3,
            name="Boundary_2",
            dim=2,
            cells=np.asarray(ym, dtype=int),
            cell_type="quad",
            label_points=np.array([[cx, y_min, cz]]),
        ),
        RegionMesh(
            tag=4,
            name="Boundary_3",
            dim=2,
            cells=np.asarray(yp, dtype=int),
            cell_type="quad",
            label_points=np.array([[cx, y_max, cz]]),
        ),
        RegionMesh(
            tag=5,
            name="Boundary_4",
            dim=2,
            cells=np.asarray(zm, dtype=int),
            cell_type="quad",
            label_points=np.array([[cx, cy, z_min]]),
        ),
        RegionMesh(
            tag=6,
            name="Boundary_5",
            dim=2,
            cells=np.asarray(zp, dtype=int),
            cell_type="quad",
            label_points=np.array([[cx, cy, z_max]]),
        ),
    ]
    return MeshSnapshot(
        points=points,
        topo_dim=3,
        domains=[dom],
        boundaries=boundaries,
        interfaces=[],
    )


class Calc3D:
    """Load a FEChem mesh and evaluate Dom / Bnd / Itf integrals on VTU fields."""

    def __init__(self, *args: Any) -> None:
        if len(args) == 1 and isinstance(args[0], str):
            from .viz import extract_snapshot_from_file

            self._msh_path: str | None = args[0]
            self._snap = extract_snapshot_from_file(args[0], topo_dim=3)
        elif len(args) == 9:
            (
                x_min,
                y_min,
                z_min,
                x_max,
                y_max,
                z_max,
                nx,
                ny,
                nz,
            ) = args
            self._msh_path = None
            self._snap = _snapshot_from_bounds(
                float(x_min),
                float(y_min),
                float(z_min),
                float(x_max),
                float(y_max),
                float(z_max),
                int(nx),
                int(ny),
                int(nz),
            )
        else:
            raise TypeError(
                "Calc3D(msh_path) or "
                "Calc3D(x_min, y_min, z_min, x_max, y_max, z_max, nx, ny, nz)"
            )

        self._points = np.asarray(self._snap.points, dtype=float)

        # FEChem region id = Gmsh physical tag - 1
        self._domains: dict[int, Any] = {}
        for region in self._snap.domains:
            self._domains[int(region.tag) - 1] = region

        self._boundaries: dict[int, Any] = {}
        for region in self._snap.boundaries:
            self._boundaries[int(region.tag) - 1] = region

        self._interfaces: dict[int, Any] = {}
        for region in self._snap.interfaces:
            self._interfaces[int(region.tag) - 1] = region

        self._dom_node_ids: dict[int, np.ndarray] = {}
        self._dom_global_to_local: dict[int, dict[int, int]] = {}
        self._dom_cells_local: dict[int, np.ndarray] = {}
        self._dom_points_xyz: dict[int, np.ndarray] = {}
        self._dom_cell_type: dict[int, str] = {}
        self._dom_quad: dict[int, Any] = {}

        self._vtu_match_tol = 5e-6

        for dom_id, region in self._domains.items():
            cells = np.asarray(region.cells, dtype=int)
            node_ids = np.unique(cells.ravel())
            node_ids.sort()
            g2l = {int(gid): loc for loc, gid in enumerate(node_ids)}
            cells_local = np.vectorize(g2l.__getitem__, otypes=[int])(cells)
            self._dom_node_ids[dom_id] = node_ids
            self._dom_global_to_local[dom_id] = g2l
            self._dom_cells_local[dom_id] = cells_local
            self._dom_points_xyz[dom_id] = self._points[node_ids]
            self._dom_cell_type[dom_id] = region.cell_type
            self._dom_quad[dom_id] = build_dom_quad_tables(
                self._dom_points_xyz[dom_id], cells_local, region.cell_type
            )

        self._face_adj = build_face_adjacency(
            self._dom_cells_local, self._dom_cell_type
        )
        self._field_domain: dict[int, int] = {}

    def show_mesh(self) -> None:
        """Display Dom / Bnd / Itf indices."""
        from .viz import render_snapshot

        render_snapshot(self._snap)

    def read_scl(self, dom: int, path: str) -> np.ndarray:
        """Load a nodal scalar VTU PointData ``value`` onto Dom-local order."""
        arr = self._read_vtu(dom, path)
        if arr.ndim != 1:
            raise ValueError(
                f"Expected scalar VTU for read_scl, got shape {arr.shape}"
            )
        return arr

    def read_vec(self, dom: int, path: str) -> np.ndarray:
        """Load a nodal vector VTU PointData ``value`` onto Dom-local order."""
        arr = self._read_vtu(dom, path)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"Expected vector VTU shape (n, 3) for read_vec, got {arr.shape}"
            )
        return arr

    def _read_vtu(self, dom: int, path: str) -> np.ndarray:
        if dom not in self._domains:
            raise ValueError(f"Unknown domain index {dom}")
        vtu_points, vtu_values = read_vtu_values(path)
        remapped = remap_vtu_to_domain(
            vtu_points,
            vtu_values,
            self._dom_points_xyz[dom],
            self._vtu_match_tol,
        )
        out = np.array(remapped, dtype=float, copy=True)
        self._field_domain[id(out)] = dom
        return out

    def _resolve_field_domain(self, *fields: Any, require: bool = True) -> int | None:
        found: set[int] = set()
        for field in fields:
            if np.isscalar(field) or callable(field):
                continue
            if isinstance(field, (tuple, list)) and len(field) == 3 and all(
                np.isscalar(c) for c in field
            ):
                continue
            arr = np.asarray(field)
            if id(field) in self._field_domain:
                found.add(self._field_domain[id(field)])
                continue
            if arr.ndim >= 1:
                n = arr.shape[0]
                matches = [
                    d for d, ids in self._dom_node_ids.items() if len(ids) == n
                ]
                if len(matches) == 1:
                    found.add(matches[0])
        if not found:
            if require:
                raise ValueError(
                    "Cannot determine domain for field; pass arrays from read_scl/read_vec"
                )
            return None
        if len(found) > 1:
            raise ValueError(
                f"Field arrays belong to multiple domains: {sorted(found)}"
            )
        return next(iter(found))

    def _surf_region(self, reg: int):
        if reg in self._boundaries:
            return self._boundaries[reg]
        if reg in self._interfaces:
            return self._interfaces[reg]
        raise ValueError(f"Unknown boundary/interface index {reg}")

    def _check_unk(self, dom: int, unk: np.ndarray, kind: str) -> np.ndarray:
        arr = np.asarray(unk, dtype=float)
        n = len(self._dom_node_ids[dom])
        if kind == "scl":
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(
                    f"Scalar field must have shape ({n},), got {arr.shape}"
                )
        else:
            if arr.ndim != 2 or arr.shape != (n, 3):
                raise ValueError(
                    f"Vector field must have shape ({n}, 3), got {arr.shape}"
                )
        return arr

    # --- scalar integrals -------------------------------------------------

    def volint_scl_src(self, dom: int, src: Any, unk: Any) -> float:
        """Volume integral ∫ src(u) dΩ over domain ``dom``."""
        if dom not in self._domains:
            raise ValueError(f"Unknown domain index {dom}")
        u = self._check_unk(dom, unk, "scl")
        return volint_scl_src(
            self._dom_quad[dom], self._dom_cells_local[dom], src, u
        )

    def surfint_scl_diff(self, bnd: int, diff: Any, unk: Any) -> float:
        """Surface integral ∫ -diff(u) (∇u · n) dS on boundary/interface ``bnd``."""
        region = self._surf_region(bnd)
        dom = self._resolve_field_domain(unk)
        assert dom is not None
        u = self._check_unk(dom, unk, "scl")
        return surfint_scl_diff(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            self._dom_points_xyz[dom],
            self._dom_cell_type[dom],
            self._face_adj,
            diff,
            u,
        )

    def surfint_scl_adv(self, bnd: int, wgt: Any, vel: Any, unk: Any) -> float:
        """Surface integral ∫ wgt(u) u (v · n) dS on boundary/interface ``bnd``."""
        region = self._surf_region(bnd)
        dom = self._resolve_field_domain(unk, vel)
        assert dom is not None
        u = self._check_unk(dom, unk, "scl")
        v = self._check_unk(dom, vel, "vec")
        return surfint_scl_adv(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            self._dom_points_xyz[dom],
            self._dom_cell_type[dom],
            self._face_adj,
            wgt,
            v,
            u,
        )

    # --- vector integrals -------------------------------------------------

    def volint_vec_src(self, dom: int, src: Any, fce: Any) -> np.ndarray:
        """Volume integral ∫ src(f) dΩ over domain ``dom``; returns (3,)."""
        if dom not in self._domains:
            raise ValueError(f"Unknown domain index {dom}")
        f = self._check_unk(dom, fce, "vec")
        return volint_vec_src(
            self._dom_quad[dom], self._dom_cells_local[dom], src, f
        )

    def surfint_vec_visc(self, bnd: int, visc: Any, vel: Any) -> np.ndarray:
        """Surface integral ∫ -(τ · n) dS, viscous force of fluid on ``bnd``."""
        region = self._surf_region(bnd)
        dom = self._resolve_field_domain(vel)
        assert dom is not None
        v = self._check_unk(dom, vel, "vec")
        return surfint_vec_visc(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            self._dom_points_xyz[dom],
            self._dom_cell_type[dom],
            self._face_adj,
            visc,
            v,
        )

    def surfint_vec_adv(self, bnd: int, den: Any, vel: Any) -> np.ndarray:
        """Surface integral ∫ ρ (v · n) v dS on ``bnd``."""
        region = self._surf_region(bnd)
        dom = self._resolve_field_domain(vel)
        assert dom is not None
        v = self._check_unk(dom, vel, "vec")
        return surfint_vec_adv(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            self._dom_points_xyz[dom],
            self._dom_cell_type[dom],
            self._face_adj,
            den,
            v,
        )

    def surfint_vec_pres(self, bnd: int, pres: Any) -> np.ndarray:
        """Surface integral ∫ p n dS, pressure force of fluid on ``bnd``."""
        region = self._surf_region(bnd)
        dom = self._resolve_field_domain(pres)
        assert dom is not None
        p = self._check_unk(dom, pres, "scl")
        return surfint_vec_pres(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            self._dom_points_xyz[dom],
            self._dom_cell_type[dom],
            self._face_adj,
            p,
        )
