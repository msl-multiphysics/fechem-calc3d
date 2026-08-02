use fechem_fem3d::*;
use std::fs::create_dir_all;

/// Steady-state heat equation with multiple domains.
/// Run with: `cargo run --release --example heat_multi`
///
/// Geometry:
/// Cubic domain with lower-front-left and upper-back-right corner cuboids.
///
/// Domains:
/// 0 - middle domain
/// 1 - lower-front-left cuboid (0.75 m x 0.25 m x 0.25 m)
/// 2 - upper-back-right cuboid (0.75 m x 0.25 m x 0.25 m)
///
/// Boundaries:
/// 0 - lower-front-left outer (left + front + bottom)
/// 1 - middle outer (bottom + front + right)
/// 2 - upper-back-right outer (right + back + top)
/// 3 - middle outer (top + back + left)
///
/// Interfaces:
/// 4 - lower-front-left interface (between dom_0 and dom_1)
/// 5 - upper-back-right interface (between dom_0 and dom_2)
///
/// Properties:
/// dom_0 - thermal conductivity (k = 0.5 W m-1 K-1)
/// dom_1 - thermal conductivity (k = 1.0 W m-1 K-1)
/// dom_2 - thermal conductivity (k = 1.0 W m-1 K-1)
/// dom_0 - heat source (Q = 0 W m-3)
/// dom_1 - heat source (Q = -500.0 W m-3)
/// dom_2 - heat source (Q = +500.0 W m-3)
///
/// Boundary conditions:
/// bnd_0 - temperature (T = 300 K)
/// bnd_1 - temperature (T = 300 K)
/// bnd_2 - temperature (T = 300 K)
/// bnd_3 - temperature (T = 300 K)
///
/// Interface conditions:
/// itf_4 - temperature and flux continuity
/// itf_5 - contact resistance (0.1 W m-2 K-1)
///
fn main() -> Result<(), FEChemError> {
    // output directory
    create_dir_all("examples/output_heat_multi").unwrap();

    // mesh and variables
    // new - import mesh from gmsh file
    // arguments: input_file
    let mut vars = Variables::new("examples/gmsh/gmsh_threereg.msh".to_string())?;

    // geometry
    // gmsh counts 2D and 3D physical groups from 1
    // subtract 1 to get FEChem domain and boundary indices
    // for interfaces, the order of the domains does not matter
    let dom_m = vars.add_dom(0)?;  // middle
    let dom_l = vars.add_dom(1)?;  // lower-front-left
    let dom_u = vars.add_dom(2)?;  // upper-back-right
    let bnd_l = vars.add_bnd(dom_l, 0)?;  // lower-front-left outer
    let bnd_m1 = vars.add_bnd(dom_m, 1)?;  // middle outer (bottom + front + right)
    let bnd_u = vars.add_bnd(dom_u, 2)?;  // upper-back-right outer
    let bnd_m2 = vars.add_bnd(dom_m, 3)?;  // middle outer (top + back + left)
    let itf_lm = vars.add_itf(dom_l, dom_m, 4)?;  // lower-front-left interface
    let itf_um = vars.add_itf(dom_u, dom_m, 5)?;  // upper-back-right interface

    // unknown domain scalars
    // arguments: domain, initial_value, output_file
    // initial_value - initial guess for steady-state problems; initial_value for transient problems
    // output_file - can be .csv or .vtu; if empty string, no file is written
    let temp_m = vars.add_scldom_unk(dom_m, 0.0, "examples/output_heat_multi/temp_m.vtu".to_string())?;
    let temp_l = vars.add_scldom_unk(dom_l, 0.0, "examples/output_heat_multi/temp_l.vtu".to_string())?;
    let temp_u = vars.add_scldom_unk(dom_u, 0.0, "examples/output_heat_multi/temp_u.vtu".to_string())?;

    // constant domain scalars
    // arguments: domain, value, output_file
    let cond_m = vars.add_scldom_con(dom_m, 0.5, "".to_string())?;  // thermal conductivity
    let cond_l = vars.add_scldom_con(dom_l, 1.0, "".to_string())?;  // thermal conductivity
    let cond_u = vars.add_scldom_con(dom_u, 1.0, "".to_string())?;  // thermal conductivity
    let hsrc_m = vars.add_scldom_con(dom_m, 0.0, "".to_string())?;  // heat source (positive if source; negative if sink)
    let hsrc_l = vars.add_scldom_con(dom_l, -500.0, "".to_string())?;  // heat source (positive if source; negative if sink)
    let hsrc_u = vars.add_scldom_con(dom_u, 500.0, "".to_string())?;  // heat source (positive if source; negative if sink)

    // constant boundary scalars
    // arguments: boundary, value, output_file
    let temp_bnd_l = vars.add_sclbnd_con(bnd_l, 300.0, "".to_string())?;  // temperature
    let temp_bnd_m1 = vars.add_sclbnd_con(bnd_m1, 300.0, "".to_string())?;  // temperature
    let temp_bnd_u = vars.add_sclbnd_con(bnd_u, 300.0, "".to_string())?;  // temperature
    let temp_bnd_m2 = vars.add_sclbnd_con(bnd_m2, 300.0, "".to_string())?;  // temperature

    // unknown interface scalars
    // arguments: interface, initial_value, output_file
    // lagrange multipliers are needed for continuity interfaces
    let lmd_lm = vars.add_sclitf_unk(itf_lm, 0.0, "".to_string())?;  // lagrange multiplier

    // constant interface scalars
    // arguments: interface, value, output_file
    // contact resistance is needed for contact resistance interfaces
    let hres_um = vars.add_sclitf_con(itf_um, 0.1, "".to_string())?;  // contact resistance

    // steady-state heat transfer solver
    // add_heat_dom - register domain with heat transfer
    // add_temp_bnd - register boundary with temperature
    // add_cont_itf - register continuity interface
    // add_hres_itf - register contact resistance interface
    let mut phys = SteadyHeat::new();
    phys.add_heat_dom(dom_m, temp_m, cond_m, hsrc_m);  // arguments: domain, T, k, Q
    phys.add_heat_dom(dom_l, temp_l, cond_l, hsrc_l);  // arguments: domain, T, k, Q
    phys.add_heat_dom(dom_u, temp_u, cond_u, hsrc_u);  // arguments: domain, T, k, Q
    phys.add_temp_bnd(bnd_l, temp_bnd_l);  // arguments: boundary, T
    phys.add_temp_bnd(bnd_m1, temp_bnd_m1);  // arguments: boundary, T
    phys.add_temp_bnd(bnd_u, temp_bnd_u);  // arguments: boundary, T
    phys.add_temp_bnd(bnd_m2, temp_bnd_m2);  // arguments: boundary, T
    phys.add_cont_itf(itf_lm, lmd_lm);  // arguments: interface, lagrange multiplier
    phys.add_hres_itf(itf_um, hres_um);  // arguments: interface, contact resistance

    // physics solver
    // arguments: max_iter, tol, damping_factor
    // damping_factor - between 0.0 and 1.0; lower for stability and higher for speed (if linear or nearly linear)
    // for highly non-linear problems, using a lower damping factor (e.g., 0.8-0.9) may be faster
    let linsolve = SolverLu::new(1)?;
    phys.solve(&mut vars, Box::new(linsolve), 10, 1e-3, 1.0)?;

    Ok(())
}
