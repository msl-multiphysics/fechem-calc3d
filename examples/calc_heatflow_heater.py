from fechem_calc3d import Calc3D

# load gmsh mesh
calc = Calc3D("examples/cases/gmsh_heater.msh")

# option to view the mesh
# calc.show_mesh()

# load nodal VTU files into domains (dependent variables only)
# arguments: domain index, vtu file path
T_c = calc.read_scl(0, "examples/cases/output_heatflow_heater/temp_c_0.vtu")  # returns np.array with temperature (T)
T_b = calc.read_scl(1, "examples/cases/output_heatflow_heater/temp_b_0.vtu")  # returns np.array with temperature (T)
T_t = calc.read_scl(2, "examples/cases/output_heatflow_heater/temp_t_0.vtu")  # returns np.array with temperature (T)
vel = calc.read_vec(0, "examples/cases/output_heatflow_heater/vel_0.vtu")  # returns (n, 3) np.array with velocity (v)
pres = calc.read_scl(0, "examples/cases/output_heatflow_heater/pres_0.vtu")  # returns np.array with pressure (p)

# volume integral (Q dV); Q may be a constant or a function of T
# arguments: domain index, source (float or callable), scalar field
# should work even if the heat source is constant
H_b = calc.volint_scl_src(1, 200.0, T_b)  # returns float64
H_t = calc.volint_scl_src(2, 200.0, T_t)  # returns float64
# H_b = calc.volint_scl_src(1, lambda T: 200.0, T_b)  # returns float64
print(H_b)
print(H_t)

# diffusive surface integral (-k * grad(T) . n dl); n is outward unit normal
# arguments: boundary or interface index, diffusivity (float or callable), scalar field
# should work even if the diffusivity is constant
# WARNING: diffusive fluxes are not necessarily conservative (see readme.txt).
q_itf6 = calc.surfint_scl_diff(4, 1.0, T_b)  # returns float64
# q_itf6 = calc.surfint_scl_diff(4, lambda T: 1.0, T_b)  # returns float64
print(q_itf6)

q_out_diff = calc.surfint_scl_diff(1, 0.1, T_c)  # returns float64
# q_out_diff = calc.surfint_scl_diff(1, lambda T: 0.1, T_c)  # returns float64
print(q_out_diff)

# advective surface integral ((rho * cp) * T * (v . n) dS); n is outward unit normal
# arguments: boundary or interface index, weight (float or callable), velocity, scalar field
# should work even if the weight is constant
q_out_adv = calc.surfint_scl_adv(1, 100.0, vel, T_c)  # returns float64; weight = rho * cp = 1000 * 0.1
# q_out_adv = calc.surfint_scl_adv(1, lambda T: 100.0, vel, T_c)  # returns float64
print(q_out_adv)

# viscous surface integral (-tau . n dS) on a wall; returns force of fluid on the wall
# arguments: boundary or interface index, viscosity (float or callable), velocity
# should work even if the viscosity is constant
F_visc = calc.surfint_vec_visc(4, 0.001, vel)  # returns np.array shape (3,)
# F_visc = calc.surfint_vec_visc(4, lambda v: 0.001, vel)  # returns np.array shape (3,)
print(F_visc)

# pressure surface integral (p * n dS) on a wall; returns force of fluid on the wall
# arguments: boundary or interface index, pressure field
F_pres = calc.surfint_vec_pres(4, pres)  # returns np.array shape (3,)
print(F_pres)

# convective surface integral (rho * (v . n) * v dS) on a wall; returns a 3-vector flux
# arguments: boundary or interface index, density (float or callable), velocity
# should work even if the density is constant
F_adv = calc.surfint_vec_adv(4, 1000.0, vel)  # returns np.array shape (3,)
# F_adv = calc.surfint_vec_adv(4, lambda v: 1000.0, vel)  # returns np.array shape (3,)
print(F_adv)
