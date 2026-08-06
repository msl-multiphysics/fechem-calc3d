from fechem_calc3d import Calc3D

# create cubic mesh
# arguments: x_min, y_min, z_min, x_max, y_max, z_max, num_elem_x, num_elem_y, num_elem_z
calc = Calc3D(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 20, 20, 20)

# option to view the mesh
# calc.show_mesh()

# load a nodal scalar VTU into one of the domains (dependent variable only)
# arguments: domain index, vtu file path
T_v0 = calc.read_scl(0, "examples/cases/output_heat_func/temp_0.vtu")  # returns np.array with temperature (T)

# volume integral (Q dV); Q may be a constant or a function of T
# arguments: domain index, source (float or callable), scalar field
# should work even if the heat source is constant
H_v0 = calc.volint_scl_src(0, lambda T: -(200.0 + 0.5 * T), T_v0)  # returns float64
# H_v0 = calc.volint_scl_src(0, -200.0, T_v0)  # returns float64
print(H_v0)

# diffusive surface integral (-k * grad(T) . n dl); n is outward unit normal
# arguments: boundary or interface index, diffusivity (float or callable), scalar field
# should work even if the diffusivity is constant
# WARNING: diffusive fluxes are not necessarily conservative (see readme.txt).
q_s1 = calc.surfint_scl_diff(1, lambda T: 0.1 + 0.3 * T, T_v0)  # returns float64
# q_s1 = calc.surfint_scl_diff(1, 0.1, T_v0)  # returns float64
print(q_s1)
