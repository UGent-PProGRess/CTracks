"""Top-level "CT sample" containers that tie a Trajectory (scan geometry) together with either
a simulated sample (CTSimulation, built from learnable track/shape projectors) or a real/loaded
dataset (CTScan), so both can be driven through the same forward-projection interface during
reconstruction and simulation.
"""

import torch

from ctrex.utils.datasets import CTDataset
from ctrex.sample_description import track_models as tm, shape_models as sm
from ctrex.utils.trajectories import Detector, Volume, Trajectory
from ctrex.projectors import CTModule, SequentialCTModule, CTEffects
from ctrex.optimization.torchtools import OptimModule


class CTSample:
    """Shared base for CTSimulation and CTScan: holds the compute device and the CT Trajectory
    (detector geometry plus scan path), and exposes the detector's region of interest as
    `detector_roi`."""

    def __init__(self, device: torch.device):
        self.device = device

    def _setup_trajectory(self, sino_params, detector = None, extend_fov = True):
        """Build (or reuse) a Detector from `sino_params`, wrap it in a Trajectory for this
        device, and apply `sino_params` onto that Trajectory's own parameters/buffers."""
        detector = detector or Detector(**sino_params)  # type: Detector  # in CTScan
        self.trajectory = Trajectory(detector, device = self.device, extend_fov = extend_fov)  # type: Trajectory
        self.trajectory.set_params(sino_params)
        pass

    @property
    def detector_roi(self):
        return self.trajectory.detector.roi

    @detector_roi.setter
    def detector_roi(self, roi):
        self.trajectory.detector.roi = roi


class CTSimulation(CTSample, OptimModule):
    """ Context for CTSimulationModules: sino_params for common trajectory, volume extent, aggregation functions"""
    
    def __init__(self, device, sino_params, projectors = None, extend_fov = True):
        """Set up the shared Trajectory from `sino_params`, then either wrap the given
        `projectors` (a SequentialCTModule of sample components) or start with an empty
        placeholder pipeline, ready for components to be attached via `add_projector`."""
        OptimModule.__init__(self)  # init first to assign modules like Trajectory
        CTSample.__init__(self, device)
        self._setup_trajectory(sino_params, extend_fov = extend_fov)
        if projectors is None:
            self.projectors = SequentialCTModule(CTEffects())  # Placeholder
        else:
            self.set_projectors(projectors)

        self._shortcuts = {}  # avoid verbose parameters after creation of a modular CT Simulation sequence

    def add_shortcut(self, name, path_fn):
        """Register `path_fn` so that `ct_simulation.<name>` resolves to it via `__getattr__`,
        letting callers reach a nested component (e.g. a projector's shape model) without
        spelling out the full attribute path."""
        self._shortcuts[name] = path_fn

    def __getattr__(self, name):
        """Attribute-lookup fallback: if `name` was registered with `add_shortcut`, return the
        shortcut instead of raising AttributeError."""
        if '_shortcuts' in self.__dict__:
            shortcuts = object.__getattribute__(self, "_shortcuts")
        else:
            return super().__getattr__(name)
        if name in shortcuts:
            shortcut = shortcuts[name]
            return shortcut
        else:
            return super().__getattr__(name)

    def getattr(self, name):  # avoid this and use getattr(obj, name)
        """NOTE: not currently called anywhere in src/scripts - the inline comment above already
        warns to prefer the builtin `getattr(obj, name)` instead of this method, which just
        forwards to it."""
        return getattr(self, name)

    @property
    def num_components(self):
        """Number of projector/sample components currently attached."""
        return len(list(self.projectors))

    def set_projectors(self, projectors: SequentialCTModule):
        """Replace the whole projector pipeline with `projectors` and link it to this
        instance's trajectory."""
        self.projectors = projectors
        self.projectors.set_ct_trajectory(self.trajectory)

    def add_projector(self, name, projector):
        """Attach a named projector module to the pipeline, wire it to this trajectory, and
        register a shortcut so its sample component's shape model is reachable directly as
        `ct_simulation.<name>`."""
        self.projectors.add_module(name, projector)
        projector.set_ct_trajectory(self.trajectory)
        self.add_shortcut(name, projector.sample_component.shape_model)

    def forward(self, sinogram: torch.Tensor, sampled_projections = slice(None)):
        """Simulate one forward pass: derive the trajectory views for `sampled_projections`
        and have every projector add its contribution into `sinogram`."""
        sampled_projections = self.trajectory.projection_indices(sampled_projections)
        views = self.trajectory.calc_trajectory(sampled_projections)
        sinogram = self.projectors(sinogram, views, sampled_projections)
        return sinogram

    def empty_sinogram(self, sino_type: type(CTDataset), roi = None, **kwargs):
        """Create an empty sinogram of `sino_type`, sized to this trajectory's geometry.
        Delegates to the first projector (any projector will do, since they all share the
        same trajectory/geometry)."""
        roi = roi or self.detector_roi
        empty_sinogram = self.projectors[0].empty_sinogram(self.trajectory.sino_params, sino_type=sino_type,
                                                           roi=roi, **kwargs)
        return empty_sinogram

    def init_components(self, num_particles, init_mode_tracks = tm.init_random, init_mode_shapes = sm.init_random):
        """Initialize every projector's sample component with particles, using `init_mode_tracks`/
        `init_mode_shapes`. `num_particles` is either one count applied to all components or a
        list giving a count per component. NOTE: not currently called anywhere in src/scripts."""
        num_particles = num_particles if isinstance(num_particles, list) else [num_particles] * len(self.projectors)
        for num_p, projector in zip(num_particles, self.projectors):  # type: (int, CTModule)
            projector.init_component(num_p, init_mode_tracks, init_mode_shapes)
                    
    def init_close(self, truth_ct_sample, sigs_tracks, sigs_shapes):
        """Initialize each of this instance's projector components close to the matching
        component in `truth_ct_sample` (e.g. for a reconstruction started near ground truth),
        with `sigs_tracks`/`sigs_shapes` controlling the spread. Components without their own
        `init_close` (e.g. non-learnable ones) are silently skipped."""
        assert type(truth_ct_sample) is CTSimulation
        for ci, (own_component, other_component) in enumerate(zip(self.projectors, truth_ct_sample.projectors)):
            try:
                own_component.init_close(other_component, sigs_tracks[ci], sigs_shapes[ci])
            except AttributeError:
                pass
            
    def copy(self):
        """Create a new CTSimulation sharing this one's device, sino_params and extend_fov
        setting, with an independent copy of the projector pipeline. NOTE: not currently
        called anywhere in src/scripts."""
        sino_params = self.trajectory.sino_params
        ct_sample_new = CTSimulation(self.device, sino_params, self.projectors.copy(), extend_fov=self.trajectory.extend_fov)
        return ct_sample_new

    def extent(self, sampled_projections):
        """Compute the combined bounding box (voxel-coordinate min, max) covering every
        non-sphere sample component's shape at `sampled_projections`, clamped to the
        trajectory's volume. Falls back to the full volume when there is nothing to bound
        (e.g. no particles yet)."""
        extents = []
        for projector in self.projectors:  # type: CTModule
            sample_component = projector.sample_component
            if (sample_component is None
                    or type(sample_component.shape_model) is sm.SphereShape
                    or sample_component.num_particles is None):
                continue
            centers_time = projector.centers_time(sampled_projections)
            extents.append(sample_component.shape_model.extent(centers_time, self.trajectory.voxel_size))
        if len(extents) == 0:
            return torch.zeros_like(self.trajectory.volume.vol_centre), self.trajectory.volume.vol_centre * 2
        extent_min = torch.amin(torch.stack([extent[0] for extent in extents], dim = 0), dim = 0)
        extent_max = torch.amax(torch.stack([extent[1] for extent in extents], dim = 0), dim = 0)
        extent_min = torch.clamp(extent_min, 0, None)
        volume = self.trajectory.volume  # type: Volume
        extent_max = torch.clamp(torch.minimum(extent_max, volume.vol_centre * 2), 0, None)
        return extent_min, extent_max

    def clamp_params(self):
        """Clamp every populated projector's track/shape parameters back into their valid
        ranges - used after an optimizer step to enforce constraints (e.g. bounds) that aren't
        naturally respected by gradient descent."""
        for projector in self.projectors:
            if not hasattr(projector, 'sample_component') or projector.sample_component is None:
                continue
            if projector.sample_component.num_particles is not None and projector.sample_component.num_particles > 0:
                projector.track_model.clamp_params()
                projector.shape_model.clamp_params()

    def small_state_dict(self, numel = 10000, ignore_words = ('ct_trajectory',)):
        """State dict restricted to tensors with at most `numel` elements (see
        OptimModule.small_state_dict), excluding the shared `ct_trajectory` by default since
        it isn't something you'd typically want to snapshot per sample."""
        return super().small_state_dict(numel, ignore_words)
    
    def requires_grad_(self, requires_grad = False):
        """Toggle `requires_grad` on every projector component's parameters."""
        for component in self.projectors:
            component.requires_grad_(requires_grad)

    def sample_attenuations(self, voxel_coordinates, sampled_projections):
        """For every populated sample component, evaluate its shape model's attenuation field
        at `voxel_coordinates` for `sampled_projections` and sum the contributions. Components
        whose shape model doesn't implement attenuation sampling contribute nothing (their
        `NotImplementedError` is caught).

        Returns:
            attenuations: summed per-voxel attenuation field, or None if no component
                contributed.
            shape_params: dict of sampled shape parameters merged across components (e.g. for
                visualization/debugging).
            limits: [min, max] attenuation values reported across components.
        """
        attenuations = None
        limits = [0,1]
        shape_params = {}
        for projector in self.projectors:  # type: CTModule
            sample_component = projector.sample_component
            if sample_component is None or sample_component.num_particles is None or sample_component.num_particles < 1:
                continue
            centers_time = projector.centers_time(sampled_projections)  # shape: (num_particles, num_projections, 3:xyz)
            try:
                shape_attenuations, sampled_shape_params = projector.shape_model.sample_attenuation(
                    voxel_coordinates, centers_time, self.trajectory.voxel_size, sampled_projections)
                shape_params.update(sampled_shape_params)
                shape_attenuations = shape_attenuations.sum(-2)  # sum over components, shape (Nz, Ny, Nx, N_components = 1, N_proj)
                limits[0] = min(limits[0], projector.shape_model.attenuation_min)
                limits[1] = max(limits[1], projector.shape_model.attenuation_max)
            except NotImplementedError:
                shape_attenuations = voxel_coordinates[...,0:1] * 0.  # dtype float
            if attenuations is None:
                attenuations = shape_attenuations
            else:
                attenuations += shape_attenuations
        return attenuations, shape_params, limits


class CTScan(CTSample):
    """Wraps an existing CTDataset (a real or pre-generated sinogram) with a Trajectory built
    from `sino_params`, so it exposes the same trajectory-based interface as CTSimulation
    without containing any simulated sample components of its own."""

    def __init__(self, ct_dataset: CTDataset, sino_params, extend_fov = True):
        """Build the trajectory using `ct_dataset.num_projections`, overriding whatever value
        `sino_params` may already carry, so the trajectory always matches the actual dataset."""
        super().__init__(ct_dataset.device)
        sino_params['num_projections'] = ct_dataset.num_projections
        self._setup_trajectory(sino_params, ct_dataset, extend_fov=extend_fov)
