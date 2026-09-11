"""
Shape models to compute particle-driven projections
Patch models like the spheres for small objects, for which the small angle approximation holds.
The small angle approximation assumes that the patch can be demagnified in the cone beam projection geometry
to work with demagnified coordinates (u',v') = (u/m,v/m) for the calculation of the optical depth in the sample domain.
Array models for larger shapes that require pixel-driven ray tracing.

Each ShapeModel describes the "what does the sample look like" side of a reconstructed
component; it is paired with a TrackModel (see the sibling track_models.py) that describes
"how does it move over time". A SampleComponent (sample_component.py) bundles one of each.
"""

import torch
import math
from torch import nn
from torchmetrics import NormalizedRootMeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure

from ctrex.optimization.lossFunctions import tv_isotropic, rebin_int
from ctrex.optimization import torchtools as tt

init_random = 'init_random'
init_zeros = 'init_zeros'


class ShapeModel(tt.OptimModule):
    """
    Base class for the shape/attenuation side of a reconstructed sample component.

    Subclasses fall into two families (see the module docstring): patch-based shapes
    like SphereShape/HollowSphereShape, which are small enough for the small-angle
    approximation and are projected as analytic patches, and array-based shapes like
    CylinderArray/VolumeArray (via the ArrayShape subclass), which are large enough to
    require pixel-driven ray tracing instead. Most of the methods here (optical_depths,
    sample_attenuation, shape_intersection, clamp_params, shake_params, init_close, extent,
    max_projected_size, shape_area) are abstract and only meaningful once implemented by a
    concrete subclass - this base class mainly holds the shared attenuation-range/learning-rate
    bookkeeping and the parameter-initialization lifecycle (init_shapes/_init_shapes).
    """
    def __init__(self, dims, device: torch.device, attenuation_range = None, learning_rate = None,
                 attenuation_rand_mean_std = (0.5, 0.1), **kwargs):
        super().__init__(None)
        self.dims = dims
        self.device = device
        # If attenuation_min is None or attenuation_max is None, the suggested limits are 0 and 1
        # But the attenuations are not clamped
        self.attenuation_min, self.attenuation_max = attenuation_range or (None, None)
        self.attenuation_rand_mean, self.attenuation_rand_std = attenuation_rand_mean_std
        self.bounds_particles = False
        for kw, arg in kwargs.items():
            setattr(self, kw, arg)
        self._init_learning_rate = learning_rate  # keep the raw value around
        self.learning_rate = learning_rate

    def _init_shapes(self, num_particles, init_mode=init_random, **kwargs):
        """Subclass hook: allocate and randomly (or zero-)initialize this shape's parameters
        for num_particles particles. Called by init_shapes, never directly."""
        raise NotImplementedError

    def init_shapes(self, num_particles, init_mode=init_random, **kwargs):
        """Initialize (or, if num_particles is None, simply return) this shape's parameters,
        then clamp them into their valid range and re-bind learning_rate now that the
        (previously uninitialized) parameters actually exist."""
        if num_particles is None:
            return self.shape_params
        self._init_shapes(num_particles, init_mode, **kwargs)
        self.clamp_params()
        self.learning_rate = self._init_learning_rate  # re-bind against now-existing params
        shape_params = self.shape_params
        return shape_params

    @property
    def shape_params(self):
        """This shape's learnable parameter tensors, in the order given by labels."""
        shape_params = [getattr(self, name) for name in self.labels]
        return shape_params

    def projection_grid(self, component_track, sampled_projections,
                        patch_rad = None, patch_margin = 0):
        """Subclass hook: build the local (v, u) pixel-offset grid used to evaluate a patch
        around each projected particle centre. Only meaningful for patch-based shapes."""
        raise NotImplementedError

    @property
    def num_particles(self):
        return len(self.shape_params[0])

    def shape_intersection(self, sinogram, component_track:torch.Tensor, sampled_projections):
        raise NotImplementedError

    def optical_depths(self, sinogram, projection_grid, projected_centres):
        """Subclass hook: compute the optical depth (attenuation-weighted intersection length)
        contributed by this shape at each projected pixel, to be added into sinogram."""
        raise NotImplementedError

    def sample_attenuation(self, voxel_coordinates, component_track, voxel_size, sampled_projections):
        """Subclass hook: for each given voxel coordinate, sample this shape's attenuation
        value (used by ray-tracing projectors that step along rays through the sample)."""
        raise NotImplementedError

    def in_bounds(self, location_xyz, component_track = None, shape_params = None):
        return ~torch.isnan(location_xyz)  # accept all locations
    
    @property
    def shape_area(self):
        """Subclass hook: a rough per-particle size measure used for visualization only."""
        raise NotImplementedError

    def max_projected_size(self, magnifications):
        raise NotImplementedError

    def extent(self, component_track, voxel_size):
        """Subclass hook: return the (min, max) xyz bounding-box corners, in voxel units,
        that this shape's particles occupy over their whole track."""
        raise NotImplementedError
    
    def init_close(self, shape_params, sigs):
        """Subclass hook: re-initialize this shape's parameters near an existing set of
        shape_params (e.g. from a ground-truth or previous reconstruction), perturbed by sigs."""
        raise NotImplementedError

    @property
    def limits(self):
        """The (min, max) attenuation bounds to use when no explicit attenuation_range was
        given, defaulting to (0, 1)."""
        attenuation_min = self.attenuation_min or 0
        attenuation_max = self.attenuation_max or 1
        return attenuation_min, attenuation_max
    
    def clamp_params(self):
        """Subclass hook: clamp all of this shape's parameters in place into their valid ranges
        (radii/attenuations bounds, etc). Called after initialization and after each shake."""
        raise NotImplementedError

    def clamp_attenuation(self, attenuations):
        """Clamp attenuations in place into (attenuation_min, attenuation_max), if either bound
        was set; a no-op otherwise. Shared helper used by subclasses' clamp_params."""
        if self.attenuation_min is not None or self.attenuation_max is not None:
            attenuations.data = torch.clamp(attenuations.data, min=self.attenuation_min, max = self.attenuation_max)

    @torch.no_grad()
    def shake_params(self, shake_width):
        """Subclass hook: perturb this shape's parameters in place by uniform random noise of
        the given width(s), e.g. as part of a resampling/annealing scheme."""
        raise NotImplementedError


class SphereShape(ShapeModel):
    """
    Solid, uniformly-attenuating sphere per particle - the most commonly used patch-based
    shape (e.g. for tracer particles in particle tracking reconstructions). Each particle has
    a radius and a single attenuation value, and is projected as an analytic chord-length patch
    (see optical_depths) rather than by ray tracing. HollowSphereShape extends this with a
    second, larger concentric shell.
    """
    def __init__(self, dims, device, attenuation_range = (None,)*2, rad_mean = 1, rad_std = 1, rad_range = (1, 3),
                 **kwargs):
        super().__init__(dims, device, attenuation_range, **kwargs)
        self.radii = torch.nn.parameter.UninitializedParameter()
        self.attenuations = torch.nn.parameter.UninitializedParameter()
        self.rad_mean = rad_mean
        self.rad_std = rad_std
        self.rad_range = rad_range
        self.contains_particles = False
        
    @property
    def labels(self):
        return ['radii','attenuations']
    
    @property
    def rad_range(self):
        return self.rad_min, self.rad_max
    
    @rad_range.setter
    def rad_range(self, rad_range):
        self.rad_min, self.rad_max = rad_range

    def max_size(self):
        """The per-particle radius, used by extent() to size the particles' bounding box."""
        return self.radii

    def max_projected_size(self, magnifications):
        raise NotImplementedError

    def optical_depths(self, sinogram, projection_grid, projected_centres, shape_params = None):
        """
        Compute each sphere's analytic chord-length patch (the optical depth of a sphere along
        a ray at perpendicular distance r from its centre is 2*sqrt(radius**2 - r**2)) over the
        local projection_grid, weighted by attenuation, using the small-angle approximation.

        Args:
            projected_centres: (u_particles, v_particles, mag_particles) - the particles' projected
                detector-pixel centres and magnifications, from the track model's projected positions.
            shape_params: optionally override (radii, attenuations) instead of using self.shape_params
                (used by HollowSphereShape to reuse this method for its inner and outer shells).

        Returns:
            sinogram: per-particle, per-angle intensity patch, shape (N_particles, N_angles, N_v, N_u).
            projection_offsets: (v_particles, u_particles) needed to place the patch back into the sinogram.
        """
        radii, attenuations = self.shape_params if shape_params is None else shape_params
        u_particles, v_particles, mag_particles = projected_centres
        projected_radii = (mag_particles * torch.unsqueeze(radii, 1)).unsqueeze(-1).unsqueeze(-1)
        # projected_radii shape: (N_particles, N_angles, 1, 1)
        
        v_grid, u_grid = projection_grid
        v_grid, u_grid = v_grid.unsqueeze(0).unsqueeze(0), u_grid.unsqueeze(0).unsqueeze(0)
        # 1/cm -> 1/mm
        attenuations_mm = torch.reshape(attenuations, (-1, 1, 1, 1)) / 10  # (N_particles, 1, 1, 1)
        # Todo: fine grid with sample_rate and bin
    
        intersection_lengths = 2*torch.sqrt(torch.clamp(projected_radii**2 - u_grid**2 - v_grid**2, 1e-20))
        sinogram = attenuations_mm * intersection_lengths  # shape: (N_particles, N_angles, N_v, N_u)
        projection_offsets = v_particles, u_particles
        return sinogram, projection_offsets

    def sample_attenuation(self, voxel_coordinates_xyz, shape_track, voxel_size, sampled_projections):
        """ For each voxel coordinate, sample the attenuation value """
        num_projections = len(sampled_projections)
        relative_coordinates_xyz = (voxel_coordinates_xyz.unsqueeze(-2).unsqueeze(-2)
                                    - shape_track.view(1, 1, 1, self.num_particles, num_projections, 3))
        # (num_z, num_y, num_x, num_particles, num_projections, 3)
        attenuations = (torch.linalg.norm(relative_coordinates_xyz, dim = -1) < (self.radii / voxel_size).unsqueeze(-1))
        attenuations = attenuations * self.attenuations.view(1, 1, 1, self.num_particles, num_projections)
        sampled_shape_params = dict(zip(self.labels, self.shape_params))
        return attenuations, sampled_shape_params  # shape: (num_z, num_y, num_x, num_particles, num_projections)

    def _init_shapes(self, num_particles, init_mode = init_random, **kwargs):
        """Draw radii from a normal(rad_mean, rad_std) distribution and start all particles at
        a constant attenuation (the midpoint of the allowed attenuation range)."""
        self.radii = nn.Parameter(torch.normal(mean=self.rad_mean, std = self.rad_std,
                                               size=(num_particles,), device = self.device))
        # constant attenuation
        attenuation_min, attenuation_max = self.limits
        self.attenuations = nn.Parameter(torch.full_like(self.radii, (attenuation_min + attenuation_max) / 2))
        return self.clamp_params()

    @torch.no_grad()
    def init_close(self, shape_params, sigs):
        """Re-initialize radii and attenuations by sampling normally around the given
        shape_params, with per-parameter standard deviations from sigs[-2:]."""
        (radii, attenuations) = shape_params
        self.radii = nn.Parameter(torch.normal(mean=radii.data, std = sigs[-2]))
        self.attenuations = nn.Parameter(torch.normal(mean=attenuations.data, std=sigs[-1]))
        return self.clamp_params()

    @property
    def shape_area(self):  # size for visualization
        return math.pi * self.radii**2

    def extent(self, component_track, voxel_size):
        radius_voxels = self.max_size() / voxel_size
        min_extent = torch.amin(component_track - radius_voxels.unsqueeze(1).unsqueeze(1), dim = (0, 1))
        max_extent = torch.amax(component_track + radius_voxels.unsqueeze(1).unsqueeze(1), dim = (0, 1))
        return min_extent, max_extent

    def clamp_params(self):
        """Clamp radii into (rad_min, rad_max) and attenuations into (attenuation_min, attenuation_max)."""
        self.radii.data = torch.clamp(self.radii.data, min=self.rad_min, max = self.rad_max)
        self.clamp_attenuation(self.attenuations)
        return self.radii, self.attenuations


    @tt.register_loss
    def sphere_radii_zscore(self):
        """Optimization loss penalizing radii that deviate from the rad_mean/rad_std prior,
        pulling reconstructed radii back towards the expected particle-size distribution."""
        zscore = (self.radii - self.rad_mean) / self.rad_std
        zscore = torch.abs(zscore)
        # only penalize if max_rad_dist standard deviations away
        # updated_zscore = torch.where(zscore < max_rad_dist, 0, zscore)
        updated_zscore = zscore
        loss = torch.mean(updated_zscore)
        return loss

    @torch.no_grad()
    def shake_params(self, shake_width):
        """Perturb radii and attenuations in place by independent uniform noise, with widths
        shake_width[0] and shake_width[1] respectively."""
        self.radii[:] += (torch.rand_like(self.radii[:]) * 2 - 1) * shake_width[0]
        self.attenuations[:] += (torch.rand_like(self.attenuations[:]) * 2 - 1) * shake_width[1]


class HollowSphereShape(SphereShape):
    """
    Sphere with two concentric radii/attenuations (an outer shell and an inner "hole"),
    modelled by projecting the large sphere and subtracting the projection of the small one.

    NOTE: not currently instantiated anywhere in scripts/ or src/ - appears to be a work in
    progress. init_close references self.radii1/self.attenuations1, which are never set
    anywhere (only self.radii/self.attenuations are, via _init_shapes), and optical_depths
    references self.attenuation2 (singular) where the attribute is actually self.attenuations2 -
    both would raise AttributeError if called.
    """
    def __init__(self, dims, device, attenuation_range = (None,)*2, rad_mean = 1, rad_std = 1, rad_range = (1, 3),
                 **kwargs):
        super().__init__(dims, device, attenuation_range, rad_mean, rad_std, rad_range, **kwargs)
        self.radii2 = torch.nn.parameter.UninitializedParameter()
        self.attenuations2 = torch.nn.parameter.UninitializedParameter()

    @property
    def labels(self):
        return ['radius','attenuation','radius2','attenuation2']

    def max_size(self):
        return self.radii2

    def optical_depths(self, sinogram, projection_grid,
                       projected_centres:torch.Tensor, shape_params = None):
        # assume radius2 is larger than radius for the moment
        # start with large spheres
        sinogram, projection_offsets = super().optical_depths(sinogram, projection_grid, projected_centres,
                                                              shape_params = (self.radii2, self.attenuation2))
        # subtract small concentric spheres
        sinogram, projection_offsets = super().optical_depths(sinogram, projection_grid, projected_centres,
                                                              shape_params = (self.radii, self.attenuations - self.attenuations2))
        
        return sinogram, projection_offsets

    def _init_shapes(self, num_particles, init_mode = init_random, **kwargs):
        radii_2 = torch.normal(mean=self.rad_mean, std = self.rad_std, size=(num_particles,), device = self.device)
        radii_1 = radii_2 * (0.2 + 0.6 * torch.rand(size=(num_particles,), device = self.device))  # 0.2 < ratio < 0.8
        attenuation_min, attenuation_max = self.limits
        attenuations_1 = torch.full_like(radii_1, (attenuation_min + attenuation_max) / 2)
        attenuations_2 = torch.full_like(radii_1, (attenuation_min + attenuation_max) / 2)
        self.radii = nn.Parameter(radii_1)
        self.attenuations = nn.Parameter(attenuations_1)
        self.radii2 = nn.Parameter(radii_2)
        self.attenuations2 = nn.Parameter(attenuations_2)
        return self.shape_params

    @torch.no_grad()
    def init_close(self, shape_params, sigs):
        """NOTE: references self.radii1/self.attenuations1, which do not exist on this class
        (only self.radii/self.attenuations are set, in _init_shapes) - calling this will raise
        an AttributeError. Intent: re-initialize both shells' radii/attenuations near
        shape_params, perturbed by sigs."""
        self.radii1.copy_(torch.normal(mean = shape_params[0], std = sigs[0]))
        self.attenuations1.copy_(torch.normal(mean = shape_params[1], std = sigs[1]))
        self.radii2.copy_(torch.normal(mean = shape_params[2], std = sigs[2]))
        self.attenuations2.copy_(torch.normal(mean = shape_params[3], std = sigs[3]))
        return self.clamp_params()

    def clamp_params(self):
        """Clamp both shells' radii/attenuations into their valid ranges."""
        self.radii.data = torch.clamp(self.radii.data, min=self.rad_min, max = self.rad_max)
        self.clamp_attenuation(self.attenuations)
        self.radii2.data = torch.clamp(self.radii2.data, min=self.rad_min, max = self.rad_max)
        self.clamp_attenuation(self.attenuations2)
        return self.named_parameters(recurse=False)

    def sphere_radii_zscore(self):
        """Like SphereShape.sphere_radii_zscore, but intended to penalize the outer shell's
        radius (radii2) instead of radii."""
        zscore = (self.radii2 - self.rad_mean) / self.rad_std
        zscore = torch.abs(zscore)
        updated_zscore = zscore
        loss = torch.mean(updated_zscore)
        return loss

    def shake_params(self, shake_width):
        raise NotImplementedError("Never checked shaking of hollow spheres")


class ArrayShape(ShapeModel):
    """
    Cannot use patch representation because of the size, so need to raytrace
    """
    # noinspection PyMethodOverriding
    def optical_depths(self, sinogram, projection_grid, rays, component_track_world:torch.Tensor, voxel_size):
        """
        Calculate the intersection lengths, weighted with the attenuation coefficients.
        """
        projection_offsets = (0.,0.)
        return sinogram, projection_offsets
    
    def sample_attenuation(self, voxel_coordinates, component_track, voxel_size, sampled_projections):
        raise NotImplementedError


class CylinderArray(ArrayShape):
    """
    A stack of concentric cylindrical layers (e.g. a capillary tube wall plus its contents),
    each with its own attenuation, sharing one (possibly tilted) axis per particle. dx_top/dy_top
    give the axis tilt at the top of the volume relative to straight-up; radii (one per layer,
    outermost last) are fixed at construction and not learned (see learning_rates['radii'] = 0
    below). Ray-traced like other ArrayShape subclasses, unlike the patch-based SphereShape family.
    """
    @property
    def labels(self):
        return ['attenuations', 'dx_top','dy_top']
    
    def __init__(self, dim, device, attenuation_range = (None,)*2, radii = None, tilt_std = 0,
                 tilt_range = (0,0), bounds = None, **kwargs):
        super().__init__(dim, device, attenuation_range, **kwargs)
        self.attenuations = torch.nn.UninitializedParameter()
        self.dx_top = torch.nn.UninitializedParameter()
        self.dy_top = torch.nn.UninitializedParameter()
        self.radii = torch.nn.Parameter(torch.tensor(radii) if radii is not None else None)
        self.tilt_std = tilt_std
        self.tilt_range = tilt_range
        self.volume_height = 256
        self.bounds = bounds
        self.bounds_particles = self.bounds is not None
        self.hollow = True
        self.learning_rates['radii'] = 0

    @property
    def tilt_range(self):
        return self.tilt_min, self.tilt_max
    
    @property
    def num_layers(self):
        return len(self.radii)
    
    def in_bounds(self, location_xyz, component_track = None, shape_params = None):
        """Check whether location_xyz is within self.bounds (perpendicular distance from the
        cylinder axis) of every particle in component_track, using the tilt (dx_top, dy_top)
        from shape_params to define each particle's axis."""
        axes = torch.cat([shape_params[1], shape_params[2],
                          self.volume_height / 2 + 0 * shape_params[1]], dim = -1)
        axes = (axes / torch.linalg.norm(axes, dim = -1, keepdim = True)).unsqueeze(0)
        rp = component_track - location_xyz.unsqueeze(0)  # relative_position
        # shape: (num_locations, 3->xyz)
        distances = torch.linalg.norm((rp - torch.linalg.vecdot(rp, axes).unsqueeze(-1) * axes), dim = -1)
        in_bounds = torch.all((distances < torch.tensor(self.bounds, device = location_xyz.device)), dim = 0)
        return in_bounds
    
    @tilt_range.setter
    def tilt_range(self, tilt_range):
        self.tilt_min, self.tilt_max = tilt_range

    def optical_depths(self, sinogram, projection_grid, rays, component_track_world:torch.Tensor, voxel_size):
        """
        Ray-trace through the concentric cylindrical layers, from outermost to innermost:
        for each layer, add the chord length through that layer's outer boundary weighted by its
        attenuation, then subtract the same chord length weighted by the next (inner) layer's
        attenuation before descending a level - so each layer ends up contributing only the
        annulus between it and the next one in. If self.hollow, layer 0 (the innermost, e.g. the
        sample cavity) is skipped entirely rather than contributing its own attenuation.
        """
        projection_offsets = (0.,0.)
        origins, directions = rays
        directions = directions / torch.linalg.norm(directions, dim = -1, keepdim = True)
        attenuations_mm = self.attenuations / 10  # 1/cm -> 1/mm  # (num_particles, num_layers)
        axes = torch.cat([self.dx_top, self.dy_top,
                          self.volume_height / 2 + 0 * self.dx_top], dim = -1)
        axes = axes / torch.linalg.norm(axes, dim = -1, keepdim = True)
        lateral = torch.cross(axes.unsqueeze(1).unsqueeze(1).unsqueeze(1), directions, dim = -1)
        # lateral shape: (num_shapes, num_projections, n_v, n_u, 3)
        lateral_mag = torch.norm(lateral, dim = -1, keepdim = False)
        distances = 0
        for ri, radius in enumerate(self.radii):  # units mm
            # subtract inner cylinder with new attenuation
            if ri > 0:
                sinogram[:] = sinogram[:] - torch.sum(distances * attenuations_mm[:,ri], dim = 0)
            # r is measured in units of axis
            under_root = (radius**2
                          - (torch.linalg.vecdot((component_track_world - origins).unsqueeze(-2).unsqueeze(-2),
                                                 lateral / lateral_mag.unsqueeze(-1)))**2)
            under_root = torch.clamp(under_root, min=1e-20)
            distances = (2 * torch.sqrt(under_root) / lateral_mag).swapaxes(1,2)
            # add outer cylinder with new attenuation
            # skip inner cylinder where sample is, omitting attenuations[0] completely
            if ri == 0 and self.hollow:
                continue
            sinogram[:] = sinogram[:] + torch.sum(distances * attenuations_mm[:,ri], dim = 0)
            
        return sinogram, projection_offsets

    def sample_attenuation(self, voxel_coordinates_xyz, shape_track, voxel_size, sampled_projections):
        """For each voxel coordinate, sample the attenuation value (cylindrical layers)."""

        # Cylinder axis
        axes = torch.cat([self.dx_top, self.dy_top, self.volume_height / 2 + 0 * self.dx_top], dim=-1)
        axes = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)  # (num_particles, 3)
        axes = axes[:, None, None, None, :]  # for broadcasting

        # Expand dimensions
        voxel_pos = voxel_coordinates_xyz.unsqueeze(0)  # (1, Nz, Ny, Nx, 3)
        center = shape_track[:, None, None, None, :]  # (num_particles, 1, 1, 1, 3)

        # Vector from cylinder center
        rel = voxel_pos - center  # (num_particles, Nz, Ny, Nx, 3)

        # Axial component
        axial_proj = torch.sum(rel * axes, dim=-1, keepdim=True) * axes

        # Radial component
        radial_vec = rel - axial_proj
        radial_dist = torch.linalg.norm(radial_vec, dim=-1)  # (num_particles, Nz, Ny, Nx)

        # Initialize attenuations
        attenuations = torch.zeros_like(radial_dist)

        # Assign layer-wise attenuation
        prev_radius = 0
        for ri, radius in enumerate(self.radii):
            radius_vx = radius / voxel_size
            mask = (radial_dist <= radius_vx) & (radial_dist > prev_radius)
            attenuations = attenuations + mask * self.attenuations[:, ri][:, None, None, None]
            prev_radius = radius_vx

        # Match your expected output shape
        attenuations = attenuations.squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)
        # (Nz, Ny, Nx, num_particles=1, num_projections=1)

        sampled_shape_params = dict(zip(self.labels, (attenuations,)))

        return attenuations, sampled_shape_params

    def _init_shapes(self, num_particles, init_mode = init_random, **kwargs):
        """Draw each layer's attenuation uniformly within the allowed attenuation range, and the
        axis tilt (dx_top, dy_top) from a normal(0, tilt_std) distribution."""
        attenuation_min, attenuation_max = self.limits
        attenuations = attenuation_min + (torch.rand((num_particles, self.num_layers), device = self.device)
                                          * (attenuation_max - attenuation_min))
        dx_top = torch.normal(0, self.tilt_std, (num_particles, 1), dtype = torch.float32, device = self.device)
        dy_top = torch.normal(0, self.tilt_std, (num_particles, 1), dtype = torch.float32, device = self.device)
        self.attenuations = nn.Parameter(attenuations)
        self.dx_top = nn.Parameter(dx_top)
        self.dy_top = nn.Parameter(dy_top)
        return self.shape_params

    def init_close(self, shape_params, sigs):
        """Re-initialize attenuations and axis tilt by sampling normally around the given
        shape_params, with per-parameter standard deviations from sigs[:3]."""
        attenuations, dx_top, dy_top = shape_params
        recon_attenuations = torch.normal(mean = attenuations, std = sigs[0])
        recon_dx_top = torch.normal(mean = dx_top, std = sigs[1])
        recon_dy_top = torch.normal(mean = dy_top, std = sigs[2])
        recon_shapes = [recon_attenuations, recon_dx_top, recon_dy_top]
        recon_shapes = self.clamp_params(recon_shapes)
        return recon_shapes

    def extent(self, component_track, voxel_size):
        """Bounding box based on the outermost layer's radius and the axis tilt at the top of
        the volume; volume_height/2 bounds the axial (z) extent."""
        device = component_track.device
        radius = self.radii[-1] / voxel_size  # mm to voxels
        corner = torch.tensor([self.dx_top + radius, self.dy_top + radius,self.volume_height / 2],
                              device = device)
        extent_min = torch.amin(component_track - corner.unsqueeze(0).unsqueeze(0), dim = (0, 1))
        extent_max = torch.amax(component_track + corner.unsqueeze(0).unsqueeze(0), dim = (0, 1))
        return extent_min, extent_max

    @torch.no_grad()
    def clamp_params(self):
        self.clamp_attenuation(self.attenuations)
        torch.clamp_(self.dx_top, min=self.tilt_min, max = self.tilt_max)
        torch.clamp_(self.dy_top, min=self.tilt_min, max = self.tilt_max)
        return self.shape_params


class VolumeArray(ArrayShape):
    """
    A single dense 3D attenuation volume (voxel grid), used for matrix-style reconstructions
    of one large object rather than many small particles (contrast with SphereShape/CylinderArray).
    Ray-traced by sampling the volume via grid_sample rather than analytic geometry. Subclassed
    by dyrect's EventVolumeArray for its own purposes.
    """
    @property
    def labels(self):
        return ['attenuations']
    
    def __init__(self, dims, device, shape: tuple[int, int, int], voxel_scale, **kwargs):
        super().__init__(dims, device, **kwargs)
        self.dim = dims
        self.num_volumes = 2  # volume 0, volume 1, time 0-1
        self.shape = shape
        self.voxel_scale = voxel_scale
        self.interpolation_mode = 'bilinear'
        self.attenuations = nn.parameter.UninitializedParameter()  # shape: (num_particles, depth, height, width)

    @property
    def depth(self):
        return int(self.shape[0])

    @property
    def height(self):
        return int(self.shape[1])

    @property
    def width(self):
        return int(self.shape[2])

    def _init_shapes(self, num_particles, init_mode = init_zeros, **kwargs):
        """Initialize the attenuation volume to all zeros (default), or, for any other
        init_mode, to normal(attenuation_rand_mean, attenuation_rand_std) noise."""
        if init_mode == init_zeros:  # constant attenuation of zero
            attenuations = torch.zeros((num_particles,) + self.shape, device = self.device)
        else:
            attenuations = torch.normal(self.attenuation_rand_mean, self.attenuation_rand_std,
                                        size = (num_particles,) + self.shape, device = self.device)
        self.attenuations = nn.Parameter(attenuations)
        return self.shape_params

    @torch.no_grad()
    def init_close(self, attenuations: torch.Tensor, sigs):
        """Re-initialize the attenuation volume near the given attenuations tensor, perturbed by
        uniform noise of width sigs. Adds a particle dimension if attenuations is a bare 3D volume."""
        attenuations = torch.as_tensor(attenuations, device = self.device, dtype = self.attenuations.dtype)
        if attenuations.ndim == 3:
            attenuations = attenuations.unsqueeze(0)  # nparticle dimension
        self.attenuations = nn.Parameter(attenuations + torch.rand(attenuations.shape, device = self.device) * sigs)
        return self.clamp_params()

    def extent(self, shape_track, voxel_size):
        """Bounding box of the whole voxel volume (half the volume's shape, in physical units)
        centred on shape_track - unlike the particle shapes, this doesn't shrink/grow per particle."""
        box_rad_voxels = (torch.as_tensor(self.shape[::-1], device = shape_track.device) / 2 * self.voxel_scale)  # xyz
        min_extent = torch.amin(shape_track - box_rad_voxels.unsqueeze(0).unsqueeze(0), dim = (0, 1))
        max_extent = torch.amax(shape_track + box_rad_voxels.unsqueeze(0).unsqueeze(0), dim = (0, 1))
        return min_extent, max_extent

    @property
    def shape_area(self):  # size for visualization
        sz, sy, sx = self.shape
        return (sx**2 + sy**2 + sz**2) * self.voxel_scale

    def clamp_params(self):
        """Clamp the attenuation volume into (attenuation_min, attenuation_max)."""
        self.clamp_attenuation(self.attenuations)
        return self.shape_params

    def optical_depths(self, sinogram, projection_grid, rays, component_track_world:torch.Tensor, voxel_size):
        """ For each ray, integrate intersection lengths through each individual voxel multiplied with attenuation """
        raise NotImplementedError

    @tt.register_loss
    def total_variation(self):
        """Isotropic total-variation regularization loss on the attenuation volume, encouraging
        piecewise-smooth reconstructions."""
        tv_field = tv_isotropic(self.attenuations, self.voxel_scale)
        tv = tv_field.sum()
        return tv

    @tt.register_loss
    def negative_attenuation(self):
        """Penalizes negative attenuation values, since attenuation is physically non-negative."""
        negative_only = torch.relu(- self.attenuations)
        return (negative_only**2).sum()

    @tt.register_loss
    def circ_loss(self, radius = None):
        """Penalizes attenuation outside a cylindrical region of the given radius (default: half
        the volume width), applying the same circular xy-mask to every z-slice of the first
        volume - used to suppress reconstruction artifacts outside a known sample/capillary boundary."""
        if radius is None:
            radius = self.width / 2
        mgrid = torch.meshgrid(torch.arange(0, self.height) - self.height / 2,
                               torch.arange(0, self.width) - self.width / 2, indexing='ij')
        mask = 1. * ((mgrid[0] ** 2 + mgrid[1] ** 2) > radius ** 2)
        loss = ((self.attenuations[0] * mask.unsqueeze(0).to(self.device)) ** 2).sum()
        return loss

    @tt.register_loss
    def tikhonov(self):
        """L2 (Tikhonov) regularization loss on the attenuation volume, penalizing large values."""
        return (self.attenuations**2).sum()


    def roi_offsets(self, shape_track):
        """Offset from each particle's track position to the corner of its region-of-interest
        volume (i.e. how far the volume's centre sits from the origin of its own voxel grid),
        used by sample_attenuation to convert world/track coordinates into volume-local ones."""
        shape_centre = torch.as_tensor(self.shape[::-1], device=self.device) / 2 * self.voxel_scale  # xyz
        roi_offsets = shape_track - shape_centre.unsqueeze(0)
        return roi_offsets

    def sample_attenuation(self, voxel_coordinates_xyz, shape_track, voxel_size, sampled_projections):
        """ For each voxel coordinate, sample the attenuation value """
        # input: (N=1 num_shapes, C=1, D_in, H_in, W_in),
        # grid: (N=1, D_out = num_projections, H_out = V, W_out = U, 3:zyx)
        # output: (N=1, C=1, D_out = num_projections, H_out = V, W_out = U
        # https://stackoverflow.com/questions/61570727/how-to-use-pytorchs-grid-sample
        # roi offset at start for shape 0
        # Fixme: check roi offset value of -1, this should probably be 0 in standard cases without ROI
        volume_index = 0
        roi_offsets = self.roi_offsets(shape_track)
        voxel_coordinates_xyz = voxel_coordinates_xyz - roi_offsets[volume_index]
        voxel_coordinates_xyz = (voxel_coordinates_xyz / self.voxel_scale
                                 / ((torch.as_tensor(self.shape[::-1], device = self.device) - 1) / 2)  # between 0 and 2
                                 - 1  # between -1 and +1 is in bounds
                                 )
        attenuations = torch.nn.functional.grid_sample(self.attenuations[volume_index].unsqueeze(0).unsqueeze(0),
                                                       voxel_coordinates_xyz.unsqueeze(0),
                                                       align_corners=True, mode=self.interpolation_mode,
                                                       padding_mode='zeros')
        attenuations = attenuations.squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)
        sampled_shape_params = dict(zip(self.labels, (attenuations,)))
        return attenuations, sampled_shape_params  # shape: (num_z, num_y, num_x, num_particles = 1, num_projections = 1)

    def scale_parameter(self, other_shape, param):
        """Rebin other_shape's named parameter (e.g. another VolumeArray's attenuations, typically
        a ground-truth volume at a finer voxel_scale) down to this shape's voxel_scale, by
        integer-factor binning, so the two can be compared voxel-for-voxel (see gt_similarity)."""
        binning = int(self.voxel_scale / other_shape.voxel_scale)
        shape_param = other_shape.get_parameter(param).detach().cpu().numpy()
        if binning == 1:
            return torch.as_tensor(shape_param)
        ims = integer_mul_shape = [int(s // binning * binning) for s in shape_param.shape]
        scaled_param = torch.as_tensor(rebin_int(shape_param[:ims[0], :ims[1], :ims[2]], (binning,)*len(ims)),
                                       device = self.device, dtype = self.get_parameter(param).dtype)
        return scaled_param

    def in_bounds(self, location_xyz, shape_track = None, shape_params = None):
        return ~torch.isnan(location_xyz)  # accept all locations

    # @tt.register_metric
    def gt_similarity(self, shape_model_gt):
        """NOTE: not currently registered as a metric (the @tt.register_metric decorator above
        is commented out) and not called anywhere else - appears to be disabled/unused for now.
        Intent: compare this volume's parameters against a ground-truth shape_model_gt (rebinned
        to a matching voxel_scale via scale_parameter) using NRMSE and SSIM, per labelled parameter."""
        if shape_model_gt is None:
            return None
        metrics = (NormalizedRootMeanSquaredError(), StructuralSimilarityIndexMeasure())
        metric_names = ('mse', 'ssim', 'mutual_info')  # Todo: add mutual_info
        metrics_dict = {f"{metric_name}_{param}":
                            metric(
                                self.get_parameter(param).detach().cpu().squeeze().unsqueeze(0).unsqueeze(0),
                                self.scale_parameter(shape_model_gt, param).cpu().squeeze().unsqueeze(0).unsqueeze(0)
                            ).cpu().numpy().item()
                        for metric_name, metric in zip(metric_names, metrics)
                        for param in self.labels}

        return metrics_dict
