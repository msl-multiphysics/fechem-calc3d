from fechem_calc3d import Calc3D

# load gmsh mesh
calc = Calc3D("examples/cases/gmsh_lid.msh")

# option to view the mesh
# calc.show_mesh()

# load nodal VTU files into one of the domains (dependent variables only)
# arguments: domain index, vtu file path
vel = calc.read_vec(0, "examples/cases/output_flow_lid/vel_0.vtu")  # returns (n, 3) np.array with velocity (v)
pres = calc.read_scl(0, "examples/cases/output_flow_lid/pres_0.vtu")  # returns np.array with pressure (p)

# force on the top lid (boundary 5)
# viscous contribution (-tau . n dS); returns force of fluid on the lid
# arguments: boundary or interface index, viscosity (float or callable), velocity
# should work even if the viscosity is constant
F_visc = calc.surfint_vec_visc(5, 0.001, vel)  # returns np.array shape (3,)
# F_visc = calc.surfint_vec_visc(5, lambda v: 0.001, vel)  # returns np.array shape (3,)
print(F_visc)

# pressure contribution (p * n dS); returns force of fluid on the lid
# arguments: boundary or interface index, pressure field
F_pres = calc.surfint_vec_pres(5, pres)  # returns np.array shape (3,)
print(F_pres)

# convective contribution (rho * (v . n) * v dS); returns a 3-vector flux
# arguments: boundary or interface index, density (float or callable), velocity
# should work even if the density is constant
F_adv = calc.surfint_vec_adv(5, 1000.0, vel)  # returns np.array shape (3,)
# F_adv = calc.surfint_vec_adv(5, lambda v: 1000.0, vel)  # returns np.array shape (3,)
print(F_adv)

# total hydrodynamic force of the fluid on the top lid
F_lid = F_visc + F_pres + F_adv  # returns np.array shape (3,)
print(F_lid)
