from devito import Eq, Operator, Function, TimeFunction, Inc, solve, sign, grad
from devito.symbolics import retrieve_functions, INT, retrieve_derivatives


def freesurface(model, eq):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation.

    Parameters
    ----------
    model : Model
        Physical model.
    eq : Eq
        Time-stepping stencil (time update) to mirror at the freesurface.
    """
    lhs, rhs = eq.args
    # Get vertical dimension and corresponding subdimension
    fsdomain = model.grid.subdomains['fsdomain']
    zfs = fsdomain.dimensions[-1]
    z = zfs.parent

    # Retrieve vertical derivatives
    dzs = {d for d in retrieve_derivatives(rhs) if z in d.dims}
    # Remove inner duplicate
    dzs = dzs - {d for D in dzs for d in retrieve_derivatives(D.expr) if z in d.dims}
    dzs = {d: d._eval_at(lhs).evaluate for d in dzs}

    # Finally get functions for evaluated derivatives
    funcs = {f for f in retrieve_functions(dzs.values())}

    mapper = {}
    # Antisymmetric mirror at negative indices
    # TODO: Make a proper "mirror_indices" tool function
    for f in funcs:
        zind = f.indices[-1]
        if (zind - z).as_coeff_Mul()[0] < 0:
            s = sign(zind.subs({z: zfs, z.spacing: 1}))
            mapper.update({f: s * f.subs({zind: INT(abs(zind))})})

    # Mapper for vertical derivatives
    dzmapper = {d: v.subs(mapper) for d, v in dzs.items()}

    fs_eq = [eq.func(lhs, rhs.subs(dzmapper), subdomain=fsdomain)]
    fs_eq.append(eq.func(lhs._subs(z, 0), 0, subdomain=fsdomain))

    return fs_eq


def laplacian(field, model, kernel):
    """
    Spatial discretization for the isotropic acoustic wave equation. For a 4th
    order in time formulation, the 4th order time derivative is replaced by a
    double laplacian:
    H = (laplacian + s**2/12 laplacian(1/m*laplacian))

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    """
    if kernel not in ['OT2', 'OT4']:
        raise ValueError("Unrecognized kernel")
    s = model.grid.time_dim.spacing
    biharmonic = field.biharmonic(1/model.m) if kernel == 'OT4' else 0
    return field.laplace + s**2/12 * biharmonic


def iso_stencil(field, model, kernel, phi1, phi2, **kwargs):
    """
    Stencil for the acoustic isotropic wave-equation:
    u.dt2 - H + damp*u.dt = 0.

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    q : TimeFunction, Function or float
        Full-space/time source of the wave-equation.
    forward : bool, optional
        Whether to propagate forward (True) or backward (False) in time.
    """
    # Forward or backward
    forward = kwargs.get('forward', True)
    if forward:
        unext = field.forward
        phi1_next = phi1.forward
        phi2_next = phi2.forward
        u_curr = field
        u_dt = field.dt
        u_dt2 = field.dt2
        phi1_t = phi1.dt
        phi2_t = phi2.dt
    else:
        unext = field.backward
        phi1_next = phi1.backward
        phi2_next = phi2.backward
        u_dt = field.dt.T
        u_dt2 = field.dt2.T
        phi1_t = phi1.dt.T
        phi2_t = phi2.dt.T

    # Get the spacial FD
    lap = laplacian(field, model, kernel)
    # Get source
    q = kwargs.get('q', 0)
    if model.pml:
        print("We are using the pml solver")
        vp = model.vp
        dampx0 = model.dampx0
        dampz0 = model.dampz0

        gphi1 = grad(phi1)
        gphi2 = grad(phi2)
        
        # Define PDE and update rule
        eq_phys = Eq(unext, solve(field.dt2 - vp * vp * lap - q, unext), subdomain=model.grid.subdomains['physdomain'])

        if model.fs:
            subds = ['pml_left','pml_right','pml_bottom']
        else:
            subds = ['pml_left','pml_right','pml_bottom','pml_top']

        # 2) PDE in PML domain
        lhs02  = u_dt2 + (dampx0+dampz0)*u_dt + (dampx0*dampz0)*field - vp * vp * lap - gphi1[0] - gphi2[1] - q 

        stencil02 = [Eq(unext, solve(lhs02, unext),
                    subdomain=model.grid.subdomains[dname])
                    for dname in subds]
        # 3) PDE for phi1
        # Example:
        grad_p = grad(u_curr)
        gu0 = grad_p[0]
        gu1 = grad_p[1]
        lhs1 = phi1_t + dampx0*phi1 - vp * vp * (dampz0 - dampx0) * gu0
        stencil1 = [
            Eq(phi1_next,
               solve(lhs1, phi1_next),
               subdomain=model.grid.subdomains[dname])
            for dname in subds]

        lhs2 = phi2_t + dampz0*phi2 - vp * vp * (dampx0 - dampz0) * gu1
        stencil2 = [
            Eq(phi2_next,
               solve(lhs2, phi2_next),
               subdomain=model.grid.subdomains[dname])
            for dname in subds
        ]
        eqns = [eq_phys, stencil02, stencil1, stencil2]

        if model.fs:
            eqns += freesurface(model, eq_phys)

        return eqns


    else:
        print("We are using the damping solver")
        eq_time = solve(model.m * field.dt2 - lap - q + model.damp * u_dt, unext)

        # Time-stepping stencil.
        eqns = [Eq(unext, eq_time, subdomain=model.grid.subdomains['physdomain'])]

        # Add free surface
        if model.fs:
            eqns.append(freesurface(model, Eq(unext, eq_time)))
        return eqns


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='OT2', **kwargs):
    """
    Construct a forward modelling operator in an acoustic medium.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    m = model.m

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    src = geometry.src
    rec = geometry.rec

    # Time-stepping equation
    phi1 = TimeFunction(name='phi1', grid=model.grid,
                        time_order=2, space_order=model.space_order,
                        save=None)
    phi2 = TimeFunction(name='phi2', grid=model.grid,
                        time_order=2, space_order=model.space_order,
                        save=None) 

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(u, model, kernel, phi1, phi2)

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src * s**2 )

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


def AdjointOperator(model, geometry, space_order=4,
                    kernel='OT2', **kwargs):
    """
    Construct an adjoint modelling operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    m = model.m

    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)

    phi1 = TimeFunction(name='phi1', grid=model.grid,
                        time_order=2, space_order=model.space_order,
                        save=None)
    phi2 = TimeFunction(name='phi2', grid=model.grid,
                        time_order=2, space_order=model.space_order,
                        save=None) 

    srca = geometry.new_src(name='srca', src_type=None)
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, phi1, phi2, forward=False)

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, expr=rec * s**2 )

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + source_a, subs=model.spacing_map,
                    name='Adjoint', **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True,
                     kernel='OT2', **kwargs):
    """
    Construct a gradient operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Option to store the entire (unrolled) wavefield.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m = model.m

    # Gradient symbol and wavefield symbols
    grad = Function(name='grad', grid=model.grid)
    u = TimeFunction(name='u', grid=model.grid, save=geometry.nt if save
                     else None, time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)

    Phi1v = TimeFunction(name='Phi1v', grid=model.grid, time_order=2,
                         space_order=space_order)
    Phi2v = TimeFunction(name='Phi2v', grid=model.grid, time_order=2,
                         space_order=space_order)

    rec = geometry.rec

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, Phi1v, Phi2v, forward=False)

    if kernel == 'OT2':
        gradient_update = Inc(grad, - u * v.dt2)
    elif kernel == 'OT4':
        gradient_update = Inc(grad, - u * v.dt2 - s**2 / 12.0 * u.biharmonic(m**(-2)) * v)
    # Add expression for receiver injection
    receivers = rec.inject(field=v.backward, expr=rec * s**2)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + [gradient_update], subs=model.spacing_map,
                    name='Gradient', **kwargs)


def BornOperator(model, geometry, space_order=4,
                 kernel='OT2', **kwargs):
    """
    Construct an Linearized Born operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m = model.m

    # Create source and receiver symbols
    src = geometry.src
    rec = geometry.rec

    # Create wavefields and a dm field
    u = TimeFunction(name="u", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    
    Phi1u = TimeFunction(name='Phi1u', grid=model.grid, time_order=1,
                         space_order=space_order)
    Phi2u = TimeFunction(name='Phi2u', grid=model.grid, time_order=1,
                         space_order=space_order)

    Phi1U = TimeFunction(name='Phi1U', grid=model.grid, time_order=1,
                         space_order=space_order)
    Phi2U = TimeFunction(name='Phi2U', grid=model.grid, time_order=1,
                         space_order=space_order)

    dm = Function(name="dm", grid=model.grid, space_order=0)

    s = model.grid.stepping_dim.spacing
    eqn1 = iso_stencil(u, model, kernel,Phi1u, Phi2u)
    eqn2 = iso_stencil(U, model, kernel,Phi1u, Phi2u, q=-dm*u.dt2)

    # Add source term expression for u
    source = src.inject(field=u.forward, expr=src * s**2)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + source + eqn2 + receivers, subs=model.spacing_map,
                    name='Born', **kwargs)
