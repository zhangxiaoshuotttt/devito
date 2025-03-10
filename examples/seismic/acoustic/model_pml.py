import numpy as np
from sympy import finite_diff_weights as fd_w
try:
    import pytest
except:
    pass

from devito import (Grid, SubDomain, Function, Constant, warning,
                    SubDimension, Eq, Inc, Operator, div, sin, Abs)
from devito.builtins import initialize_function, gaussian_smooth, mmax, mmin
from devito.tools import as_tuple

__all__ = ['SeismicModel', 'Model', 'ModelElastic',
           'ModelViscoelastic', 'ModelViscoacoustic']


def initialize_damp(damp, padsizes, spacing, abc_type="damp", fs=False):
    """
    Initialize damping field with an absorbing boundary layer.

    Parameters
    ----------
    damp : Function
        The damping field for absorbing boundary condition.
    nbl : int
        Number of points in the damping layer.
    spacing :
        Grid spacing coefficient.
    mask : bool, optional
        whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    """

    eqs = [Eq(damp, 1.0 if abc_type == "mask" else 0.0)]
    for (nbl, nbr), d in zip(padsizes, damp.dimensions):
        if not fs or d is not damp.dimensions[-1]:
            dampcoeff = 1.5 * np.log(1.0 / 0.001) / (nbl)
            # left
            dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                      thickness=nbl)
            pos = Abs((nbl - (dim_l - d.symbolic_min) + 1) / float(nbl))
            val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
            val = -val if abc_type == "mask" else val
            eqs += [Inc(damp.subs({d: dim_l}), val/d.spacing)]
        # right
        dampcoeff = 1.5 * np.log(1.0 / 0.001) / (nbr)
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=nbr)
        pos = Abs((nbr - (d.symbolic_max - dim_r) + 1) / float(nbr))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if abc_type == "mask" else val
        eqs += [Inc(damp.subs({d: dim_r}), val/d.spacing)]

    Operator(eqs, name='initdamp')()


class PhysicalDomain(SubDomain):

    name = 'physdomain'

    def __init__(self, so, fs=False):
        super().__init__()
        self.so = so
        self.fs = fs

    def define(self, dimensions):
        map_d = {d: d for d in dimensions}
        if self.fs:
            map_d[dimensions[-1]] = ('middle', self.so, 0)
        return map_d

class PMLLeft(SubDomain):
    name = 'pml_left'
    def __init__(self,so, npml=20):
        super().__init__()
        self.so = so
        self.npml = npml
    def define(self, dimensions):
        x, z = dimensions
        return {x: ('left', self.npml), z: z}

class PMLRight(SubDomain):
    name = 'pml_right'
    def __init__(self,so, npml=20):
        super().__init__()
        self.so = so
        self.npml = npml
    def define(self, dimensions):
        x, z = dimensions
        return {x: ('right', self.npml), z: z}

class PMLBottom(SubDomain):
    name = 'pml_bottom'
    def __init__(self,so, npml=20):
        super().__init__()
        self.so = so
        self.npml = npml
    def define(self, dimensions):
        x, z = dimensions
        return {x: ('middle', self.npml, self.npml),
                z: ('left', self.npml)}

class PMLTop(SubDomain):
    name = 'pml_top'
    def __init__(self,so, fs=False, npml=20):
        super().__init__()
        self.so = so
        self.fs = fs
        self.npml = npml
    def define(self, dimensions):
        if self.fs:
            return {}  # No top PML when free surface is enabled
        x, z = dimensions
        return {x: ('middle', self.npml, self.npml),
                z: ('right', self.npml)}


class FSDomain(SubDomain):

    name = 'fsdomain'

    def __init__(self, so):
        super().__init__()
        self.size = so

    def define(self, dimensions):
        """
        Definition of the upper section of the domain for wrapped indices FS.
        """

        return {d: (d if not d == dimensions[-1] else ('left', self.size))
                for d in dimensions}


class GenericModel:
    """
    General model class with common properties
    """
    def __init__(self, origin, spacing, shape, space_order, nbl=20,
                 dtype=np.float32, subdomains=(), bcs="damp", grid=None,
                 fs=False, pml=True, topology=None):
        self.shape = shape
        self.space_order = space_order
        self.nbl = int(nbl)
        self.origin = tuple([dtype(o) for o in origin])
        self.fs = fs
        # Default setup
        origin_pml = [dtype(o - s*nbl) for o, s in zip(origin, spacing)]
        shape_pml = np.array(shape) + 2 * self.nbl

        # Model size depending on freesurface
        physdomain = PhysicalDomain(space_order, fs=fs)
        subdomains = subdomains + (physdomain,)

        if bcs == "pml":
            # Only setup PML subdomains
            PMLLeftdomain = PMLLeft(space_order,npml=self.nbl)
            PMLRightdomain = PMLRight(space_order,npml=self.nbl)
            PMLBottomdomain = PMLBottom(space_order,npml=self.nbl)
            subdomains = subdomains + (PMLLeftdomain, PMLRightdomain, PMLBottomdomain,)
            if not self.fs:
                PMLTopdomain = PMLTop(space_order,fs=self.fs, npml=self.nbl)
                subdomains = subdomains + (PMLTopdomain,)
        else:
            pass

        if fs:
            fsdomain = FSDomain(space_order)
            subdomains = subdomains + (fsdomain,)
            origin_pml[-1] = origin[-1]
            shape_pml[-1] -= self.nbl

        # Origin of the computational domain with boundary to inject/interpolate
        # at the correct index
        if grid is None:
            # Physical extent is calculated per cell, so shape - 1
            extent = tuple(np.array(spacing) * (shape_pml - 1))
            self.grid = Grid(extent=extent, shape=shape_pml, origin=origin_pml,
                             dtype=dtype, subdomains=subdomains, topology=topology)
        else:
            self.grid = grid

        self._physical_parameters = set()
        self.damp = None

        if bcs == 'damp' or bcs == 'mask':
            print("Now using simple damping")
            self._initialize_bcs(bcs)
        

    def _initialize_bcs(self, bcs="damp"):
        # Create dampening field as symbol `damp`
        if self.nbl == 0:
            self.damp = 1 if bcs == "mask" else 0
            return

        # First initialization
        init = self.damp is None
        # Get current Function if already initialized
        self.damp = self.damp or Function(name="damp", grid=self.grid,
                                          space_order=self.space_order)
        if callable(bcs):
            bcs(self.damp, self.nbl)
        else:
            re_init = ((bcs == "mask" and mmin(self.damp) == 0) or
                       (bcs == "damp" and mmax(self.damp) == 1))
            if init or re_init:
                if re_init and not init:
                    bcs_o = "damp" if bcs == "mask" else "mask"
                    warning("Re-initializing damp profile from %s to %s" % (bcs_o, bcs))
                    warning("Model has to be created with `bcs=\"%s\"`"
                            "for this WaveSolver" % bcs)
                initialize_damp(self.damp, self.padsizes, self.spacing,
                                abc_type=bcs, fs=self.fs)
        self._physical_parameters.update(['damp'])

    @property
    def padsizes(self):
        """
        Padding size for each dimension.
        """
        padsizes = [(self.nbl, self.nbl) for _ in range(self.dim-1)]
        padsizes.append((0 if self.fs else self.nbl, self.nbl))
        return padsizes

    def physical_params(self, **kwargs):
        """
        Return all set physical parameters and update to input values if provided
        """
        known = [getattr(self, i) for i in self.physical_parameters]
        return {i.name: kwargs.get(i.name, i) or i for i in known}

    def _gen_phys_param(self, field, name, space_order, is_param=True,
                        default_value=0, avg_mode='arithmetic', **kwargs):
        if field is None:
            return default_value
        if isinstance(field, np.ndarray):
            function = Function(name=name, grid=self.grid, space_order=space_order,
                                parameter=is_param, avg_mode=avg_mode)
            initialize_function(function, field, self.padsizes)
        else:
            function = Constant(name=name, value=field, dtype=self.grid.dtype)
        self._physical_parameters.update([name])
        return function

    @property
    def physical_parameters(self):
        return as_tuple(self._physical_parameters)

    @property
    def dim(self):
        """
        Spatial dimension of the problem and model domain.
        """
        return self.grid.dim

    @property
    def spacing(self):
        """
        Grid spacing for all fields in the physical model.
        """
        return self.grid.spacing

    @property
    def space_dimensions(self):
        """
        Spatial dimensions of the grid
        """
        return self.grid.dimensions

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each `SpaceDimension`.
        """
        return self.grid.spacing_map

    @property
    def dtype(self):
        """
        Data type for all assocaited data objects.
        """
        return self.grid.dtype

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))


class SeismicModel(GenericModel):
    """
    The physical model used in seismic inversion processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    vp : array_like or float
        Velocity in km/s.
    nbl : int, optional
        The number of absorbin layers for boundary damping.
    bcs: str or callable
        Absorbing boundary type ("damp" or "mask") or initializer.
    dtype : np.float32 or np.float64
        Defaults to np.float32.
    epsilon : array_like or float, optional
        Thomsen epsilon parameter (0<epsilon<1).
    delta : array_like or float
        Thomsen delta parameter (0<delta<1), delta<epsilon.
    theta : array_like or float
        Tilt angle in radian.
    phi : array_like or float
        Asymuth angle in radian.
    b : array_like or float
        Buoyancy.
    vs : array_like or float
        S-wave velocity.
    qp : array_like or float
        P-wave attenuation.
    qs : array_like or float
        S-wave attenuation.
    """
    _known_parameters = ['vp', 'damp', 'vs', 'b', 'epsilon', 'delta',
                         'theta', 'phi', 'qp', 'qs', 'lam', 'mu']

    def __init__(self, origin, spacing, shape, space_order, vp, nbl=20, fs=False, pml=True,
                 dtype=np.float32, subdomains=(), bcs="mask", grid=None,
                 topology=None, **kwargs):
        super().__init__(origin, spacing, shape, space_order, nbl, dtype, subdomains,
                         topology=topology, grid=grid, bcs=bcs, fs=fs, pml=pml)

        if bcs == 'pml' and pml==True:
            print("Now using PML")
            self.initialize_pml_damp()
        # Initialize physics
        self._initialize_physics(vp, space_order, **kwargs)

        # User provided dt
        self._dt = kwargs.get('dt')
        # Some wave equation need a rescaled dt that can't be infered from the model
        # parameters, such as isoacoustic OT4 that can use a dt sqrt(3) larger than
        # isoacoustic OT2. This property should be set from a wavesolver or after model
        # instanciation only via model.dt_scale = value.
        self._dt_scale = 1

    def _initialize_physics(self, vp, space_order, **kwargs):
        """
        Initialize physical parameters and type of physics from inputs.
        The types of physics supported are:
        - acoustic: [vp, b]
        - elastic: [vp, vs, b] represented through Lame parameters [lam, mu, b]
        - visco-acoustic: [vp, b, qp]
        - visco-elastic: [vp, vs, b, qs]
        - vti: [vp, epsilon, delta]
        - tti: [epsilon, delta, theta, phi]
        """
        params = []
        # Buoyancy
        b = kwargs.get('b', 1)

        # Initialize elastic with Lame parametrization
        if 'vs' in kwargs:
            vs = kwargs.pop('vs')
            self.lam = self._gen_phys_param((vp**2 - 2. * vs**2)/b, 'lam', space_order,
                                            is_param=True)
            self.mu = self._gen_phys_param(vs**2 / b, 'mu', space_order, is_param=True,
                                           avg_mode='harmonic')
        else:
            # All other seismic models have at least a velocity
            self.vp = self._gen_phys_param(vp, 'vp', space_order)
        # Initialize rest of the input physical parameters
        for name in self._known_parameters:
            if kwargs.get(name) is not None:
                field = self._gen_phys_param(kwargs.get(name), name, space_order)
                setattr(self, name, field)
                params.append(name)

    def initialize_pml_damp(self, sigma_max=0.1):
        """
        Create custom PML damping Functions for each dimension.
        Must use the full padded shape (grid.shape).
        """
        grid = self.grid
        nxpad, nzpad = grid.shape  # e.g. (600,274)
        hx, hz = self.spacing

        # Each PML boundary is self.nbl cells
        lx = self.nbl * hx
        lz = self.nbl * hz

        # Create your damping Functions
        dampx0 = Function(name='dampx0', grid=grid, space_order=self.space_order)
        dampx1 = Function(name='dampx1', grid=grid, space_order=self.space_order)
        dampz0 = Function(name='dampz0', grid=grid, space_order=self.space_order)
        dampz1 = Function(name='dampz1', grid=grid, space_order=self.space_order)

        # Build coordinate arrays over the full padded domain
        x0 = grid.origin[0]
        z0 = grid.origin[1]

        x_vals = np.linspace(x0, x0 + hx*(nxpad-1), nxpad)
        z_vals = np.linspace(z0, z0 + hz*(nzpad-1), nzpad)
        X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')  # shape (nxpad, nzpad)

        def pml_profile(coord, start_pml, end_pml, L_pml):
            """
            E.g. sigma = sigma_max*(alpha - sin(2π alpha)/(2π))
            """
            alpha = np.zeros_like(coord)
            left_idx  = coord <  start_pml
            right_idx = coord >  end_pml
            alpha[left_idx]  = np.abs(coord[left_idx]  - start_pml)/L_pml
            alpha[right_idx] = np.abs(coord[right_idx] - end_pml)/L_pml
            # Smooth taper
            out = sigma_max*(alpha - np.sin(2.*np.pi*alpha)/(2.*np.pi))
            return out

        # We'll define the region of "inner domain" as [x0 + lx, x0+(nxpad-nbl-1)*hx], etc.
        # start_pml_x = x0 + lx
        # end_pml_x   = x0 + (nxpad - self.nbl - 1)*hx
        start_pml_x = x0 + lx
        end_pml_x   = x0 + (nxpad - self.nbl - 1)*hx
        start_pml_z = z0 + lz
        end_pml_z   = z0 + (nzpad - self.nbl - 1)*hz

        # Fill in arrays
        dampx0.data[:] = pml_profile(X, start_pml_x, end_pml_x, lx)
        dampx1.data[:] = pml_profile(X + 0.5*hx, start_pml_x, end_pml_x, lx)

        dampz0.data[:] = pml_profile(Z, start_pml_z, end_pml_z, lz)
        dampz1.data[:] = pml_profile(Z + 0.5*hz, start_pml_z, end_pml_z, lz)

        # Store them
        self.dampx0 = dampx0
        self.dampx1 = dampx1
        self.dampz0 = dampz0
        self.dampz1 = dampz1

    @property
    def _max_vp(self):
        if 'vp' in self._physical_parameters:
            return mmax(self.vp)
        else:
            return np.sqrt(mmin(self.b) * (mmax(self.lam) + 2 * mmax(self.mu)))

    @property
    def _thomsen_scale(self):
        # Update scale for tti
        if 'epsilon' in self._physical_parameters:
            return np.sqrt(1 + 2 * mmax(self.epsilon))
        return 1

    @property
    def dt_scale(self):
        return self._dt_scale

    @dt_scale.setter
    def dt_scale(self, val):
        self._dt_scale = val

    @property
    def _cfl_coeff(self):
        """
        Courant number from the physics and spatial discretization order.
        The CFL coefficients are described in:
        - https://doi.org/10.1137/0916052 for the elastic case
        - https://library.seg.org/doi/pdf/10.1190/1.1444605 for the acoustic case
        """
        # Elasic coefficient (see e.g )
        if 'lam' in self._physical_parameters or 'vs' in self._physical_parameters:
            coeffs = fd_w(1, range(-self.space_order//2+1, self.space_order//2+1), .5)
            c_fd = sum(np.abs(coeffs[-1][-1])) / 2
            return .95 * np.sqrt(self.dim) / self.dim / c_fd
        a1 = 4  # 2nd order in time
        coeffs = fd_w(2, range(-self.space_order, self.space_order+1), 0)[-1][-1]
        return np.sqrt(a1/float(self.grid.dim * sum(np.abs(coeffs))))

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        dt = self._cfl_coeff * np.min(self.spacing) / (self._thomsen_scale*self._max_vp)
        dt = self.dtype("%.3e" % (self.dt_scale * dt))
        if self._dt:
            return self._dt
        return dt

    def update(self, name, value):
        """
        Update the physical parameter param.
        """
        try:
            param = getattr(self, name)
        except AttributeError:
            # No physical parameter with tha name, create it
            setattr(self, name, self._gen_phys_param(value, name, self.space_order))
            return
        # Update the physical parameter according to new value
        if isinstance(value, np.ndarray):
            if value.shape == param.shape:
                param.data[:] = value[:]
            elif value.shape == self.shape:
                initialize_function(param, value, self.nbl)
            else:
                raise ValueError("Incorrect input size %s for model" % value.shape +
                                 " %s without or %s with padding" % (self.shape,
                                                                     param.shape))
        else:
            param.data = value

    @property
    def m(self):
        """
        Squared slowness.
        """
        return 1 / (self.vp * self.vp)

    @property
    def dm(self):
        """
        Create a simple model perturbation from the velocity as `dm = div(vp)`.
        """
        dm = Function(name="dm", grid=self.grid, space_order=self.space_order)
        Operator(Eq(dm, div(self.vp)), subs=self.spacing_map)()
        return dm

    def smooth(self, physical_parameters, sigma=5.0):
        """
        Apply devito.gaussian_smooth to model physical parameters.

        Parameters
        ----------
        physical_parameters : string or tuple of string
            Names of the fields to be smoothed.
        sigma : float
            Standard deviation of the smoothing operator.
        """
        model_parameters = self.physical_params()
        for i in physical_parameters:
            gaussian_smooth(model_parameters[i], sigma=sigma)
        return


@pytest.mark.parametrize('shape', [(51, 51), (16, 16, 16)])
def test_model_update(shape):

    # Set physical properties as numpy arrays
    vp = np.full(shape, 2.5, dtype=np.float32)  # vp constant of 2.5 km/s
    qp = 3.516*((vp[:]*1000.)**2.2)*10**(-6)  # Li's empirical formula
    b = 1 / (0.31*(vp[:]*1000.)**0.25)  # Gardner's relation

    # Define a simple visco-acoustic model from the physical parameters above
    va_model = SeismicModel(space_order=4, vp=vp, qp=qp, b=b, nbl=10,
                            origin=tuple([0. for _ in shape]), shape=shape,
                            spacing=[20. for _ in shape],)

    # Define a simple acoustic model with vp=1.5 km/s
    model = SeismicModel(space_order=4, vp=np.full_like(vp, 1.5), nbl=10,
                         origin=tuple([0. for _ in shape]), shape=shape,
                         spacing=[20. for _ in shape],)

    # Define a velocity Function
    vp_fcn = Function(name='vp0', grid=model.grid, space_order=4)
    vp_fcn.data[:] = 2.5

    # Test 1. Update vp of acoustic model from array
    model.update('vp', vp)
    assert np.array_equal(va_model.vp.data, model.vp.data)

    # Test 2. Update vp of acoustic model from Function
    model.update('vp', vp_fcn.data)
    assert np.array_equal(va_model.vp.data, model.vp.data)

    # Test 3. Create a new physical parameter in the acoustic model from array
    model.update('qp', qp)
    assert np.array_equal(va_model.qp.data, model.qp.data)

    # Make a set of physical parameters from each model
    tpl1_set = set(va_model.physical_parameters)
    tpl2_set = set(model.physical_parameters)

    # Physical parameters in either set but not in the intersection.
    diff_phys_par = tuple(tpl1_set ^ tpl2_set)

    # Turn acoustic model (it is just lacking 'b') into a visco-acoustic model
    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(model.dim))
    for i in diff_phys_par:
        # Test 4. Create a new physical parameter in the acoustic model from function
        model.update(i, getattr(va_model, i).data[slices])
        assert np.array_equal(getattr(model, i).data, getattr(va_model, i).data)


# For backward compatibility
Model = SeismicModel
ModelElastic = SeismicModel
ModelViscoelastic = SeismicModel
ModelViscoacoustic = SeismicModel
