"""
Track models describing the "how does it move over time" side of a reconstructed sample
component: each TrackModel maps a batch of track times to per-particle xyz positions at those
times, via a small set of learnable control points. Simpler models (StaticTrack, LinearTrack,
PiecewiseLinearTrack) interpolate directly between control points; KnownTrack replays a fixed,
non-learnable track; NurbsTrack fits a smooth NURBS curve through its control points. Paired with
a ShapeModel (see the sibling shape_models.py) describing the "what does it look like" side.
"""

import torch
from torch import nn

from ctrex.sample_description.PoreMaskFunctions import PoreMaskPlain, PoreMaskFunctions
from ctrex.sample_description.initFunctions import (random_coordinates_mask, random_coordinates_volume,
                                                    close_coordinates_mask, close_control_points,
                                                    nearby_velocity_init)
from ctrex.optimization import torchtools as tt

"""
Everything in sample domain (xyzt)
"""

init_random = 'init_random'
init_zeros = 'init_zeros'
init_random_nearby_vel = 'init_random_nearby_vel'


class TrackModel(tt.OptimModule):
    """
    Base class for the track/motion side of a reconstructed sample component: a learnable set
    of per-particle control_points (shape depends on the subclass, but always has a particle
    dimension first) that forward() turns into an xyz position for each requested track_time.
    Optionally confines particles to a pore_mask (a segmented pore-space distance map - see
    PoreMaskFunctions.py) so tracks stay physically inside the sample. Most of the concrete
    behaviour (forward, _init_tracks, init_close, clamp_params, shake_params, calc_distance_matrix)
    is left abstract here and implemented by subclasses (KnownTrack, StaticTrack and its
    LinearTrack/PiecewiseLinearTrack/NurbsTrack descendants).
    """
    def __init__(self, dims, device, vol_shape, learning_rate = None, pore_mask = None,
                 mask_confines = True, **kwargs):
        super().__init__(None)
        self.dims = dims
        self.device = device
        self.vol_shape = vol_shape
        self.pore_mask = pore_mask
        self.mask_confines = mask_confines  # indicate if pore_mask should be active for this model
        for kw, arg in kwargs.items():
            setattr(self, kw, arg)
        self.control_points = torch.nn.UninitializedParameter()
        self.learning_rate = learning_rate
    
    @property
    def num_control_points(self):
        """Subclass hook: how many control points make up one particle's track (e.g. 1 for
        StaticTrack, 2 for LinearTrack, num_segments + 1 for PiecewiseLinearTrack)."""
        raise NotImplementedError
    
    @num_control_points.setter
    def num_control_points(self, value):
        raise NotImplementedError

    @property
    def num_particles(self):
        return self.control_points.shape[0]
    
    @property
    def labels(self):
        """Subclass hook: human-readable names for each of this track's control-point
        coordinates, in the order used by control_points' last dimension."""
        raise NotImplementedError
    
    def forward(self, track_time):
        """Subclass hook: evaluate every particle's track at each given track_time (a value in
        [0, 1] along the track), returning xyz positions of shape (num_particles, len(track_time), 3)."""
        raise NotImplementedError
    
    @property
    def track_params(self):
        """The sole top-level learnable parameter tensor of this track model, i.e. control_points."""
        return list(self.parameters(False))[0]
    
    def _init_tracks(self, num_particles, init_mode = init_random):
        """Subclass hook: allocate and randomly initialize control_points for num_particles
        particles. Called by init_tracks, never directly."""
        raise NotImplementedError

    def init_tracks(self, num_particles, init_mode = init_random):
        """Initialize (or, if num_particles is None, simply return) control_points, then clamp
        them into bounds/pore-mask confines."""
        if num_particles is None:
            return self.control_points
        self._init_tracks(num_particles, init_mode)
        self.clamp_params()
        return self.control_points

    def init_close(self, track_params, sigs = None):
        """Subclass hook: re-initialize control_points near an existing set of track_params (e.g.
        from a ground-truth or previous reconstruction), perturbed by sigs."""
        raise NotImplementedError

    @torch.no_grad()
    def clamp_params(self, shape_params, part_dist_params):
        """Subclass hook: clamp control_points in place into the volume bounds and/or pore_mask
        confines. Note concrete subclasses override this with a different signature (StaticTrack/
        LinearTrack/PiecewiseLinearTrack take just self; NurbsTrack takes a track_params arg
        instead) - this base (shape_params, part_dist_params) signature is never actually used."""
        raise NotImplementedError

    @torch.no_grad()
    def shake_params(self, shake_width):
        """Subclass hook: perturb control_points in place by random noise of the given width(s)."""
        raise NotImplementedError

    @torch.no_grad()
    def append_params(self, track_params, new_track_params):
        """NOTE: not called anywhere in src/ or scripts/ - SampleComponent.extend implements
        particle-adding directly instead. Intent: concatenate new_track_params onto track_params
        along the particle dimension, preserving requires_grad."""
        appended_track_params = torch.cat((track_params, new_track_params), dim=0)
        appended_track_params.requires_grad_(track_params.requires_grad)
        return appended_track_params

    @torch.no_grad()
    def remove_params(self, track_params, removal_indices):
        """NOTE: not called anywhere in src/ or scripts/ - SampleComponent.remove_particles
        implements particle-removal directly instead. Intent: drop the particles selected by
        removal_indices (a boolean mask) from track_params, preserving requires_grad."""
        removed_params = track_params[~removal_indices]
        removed_params.requires_grad_(track_params.requires_grad)
        return removed_params

    @torch.no_grad()
    def calc_distance_matrix(self):
        """Subclass hook: pairwise distance matrix between all particles' tracks (used e.g. to
        detect near-duplicate/colliding particles)."""
        raise NotImplementedError

    def centers_time(self, track_time):
        """Convenience alias for calling this track model directly (self(track_time))."""
        centers_time = self(track_time)
        return centers_time

    @tt.register_loss
    def out_of_pore_tracks(self):
        """Optimization loss penalizing control points that lie outside (or far from) the
        pore_mask's segmented pore space, using its precomputed distance map, normalized so
        points already deep inside the pore contribute (near) zero."""
        coords_zyx_sample = self.pore_mask.get_clamped_coordinates(self.control_points[..., :3])
        distances = self.pore_mask.distance_map[coords_zyx_sample]
        normalized_distances = torch.clamp(distances / self.pore_mask.max_distance, 0.0, 1.0)
        return torch.sum(normalized_distances)

    def check_tracks_in_pore(self):
        """Per-particle boolean mask of whether every control point of that particle's track
        lies inside the pore mask (True for all particles when there is no pore_mask)."""
        if self.pore_mask is None:
            return torch.ones(self.num_particles, dtype = torch.bool, device = self.device)
        return self.pore_mask.check_particles_in_pore(self.control_points)



class KnownTrack(TrackModel):
    """
    Fixed, non-learnable track: control_points is a pre-supplied, complete per-projection
    position for each particle (requires_grad = False), used to replay a known ground-truth or
    externally-tracked trajectory (e.g. for testing a shape-only reconstruction against known
    particle positions) rather than to learn a track from data.
    """
    num_control_points = 2
    def __init__(self, dims, device, vol_shape, control_points, **kwargs):
        super().__init__(dims, device, vol_shape, **kwargs)
        self.control_points = nn.Parameter(control_points, requires_grad = False)
        # self.num_control_points = 2 #for display?

    @property
    def labels(self):
        return ['center_x', 'center_y', 'center_z']

    def forward(self, track_time):
        """Look up the nearest-matching pre-supplied position for each requested track_time,
        by rounding track_time (assumed in [0, 1]) to one of the M stored control-point indices.
        Asserts that the resulting indices are unique, since this class expects exactly one
        stored position per requested projection, not genuine interpolation."""
        track_params = self.control_points
        M = track_params.shape[1]  # Todo: give name
        track_index = (track_time*(M-1)).round().to(torch.long)  # range from 0 to max index
        assert len(torch.unique(track_index)) == len(track_index), "Non-unique indices in KnownTrack - check global angles match simulated number of projections"
        return track_params[:,track_index,:]


class StaticTrack(TrackModel):
    """
    Non-moving particles: a single control point per particle, used as-is for every track_time
    (broadcast, not interpolated). The simplest learnable track model, and the base class for
    the moving-track models LinearTrack/PiecewiseLinearTrack (which reuse its bounds-clamping,
    pore-mask-based initialization, and distance-matrix logic, but add extra control points and
    override forward/_init_tracks/init_close/clamp_params for actual motion).
    """
    num_control_points = 1

    def __init__(self, dims, device, vol_shape, bounds =(0,) * 6, **kwargs):
        self.bounds = bounds  # Todo: make this part of pore_mask?
        super().__init__(dims, device, vol_shape, **kwargs)
    
    @property
    def labels(self):
        return ['center_x', 'center_y', 'center_z']

    def _init_tracks(self, num_particles, init_mode = init_random):
        """Randomly place num_particles particles either within the pore mask (if mask_confines)
        or anywhere in the volume."""
        if self.mask_confines:
            centers_xyz = random_coordinates_mask(num_particles, self.pore_mask, self.device)
        else:
            centers_xyz = random_coordinates_volume(num_particles, self.vol_shape, self.device, self.dims)
        centers_xyz = centers_xyz.unsqueeze(1)  # add time dimension
        self.control_points = nn.Parameter(centers_xyz)
        return self.control_points

    def init_close(self, track_params, sigs = None):
        """Re-initialize the single control point near track_params' first control point,
        perturbed by sigs[:3] (via close_control_points), then clamp into bounds."""
        centers_xyz = close_control_points(track_params[:, :1, :3], self.pore_mask, sigs[:3])
        self.control_points = nn.Parameter(centers_xyz)
        self.clamp_params()
        return centers_xyz

    def forward(self, track_time):
        """Return the same (single) control point position for every requested track_time,
        confining into the pore mask if one is set and active."""
        offsets = self.control_points[:]  # .unsqueeze(1) - choosing to unsqueeze from the start  # (N_particles, 1, 3)
        tracks = offsets.to(track_time.device) + track_time.view(1,-1,1) * 0
        if self.pore_mask is not None and self.mask_confines:  # confine particles
            tracks = self.pore_mask.move_coords_to_pore(tracks)
        return tracks  # (N_particles, N_theta, 3)

    @torch.no_grad()
    def clamp_params(self):
        """Clamp each xyz coordinate of control_points into [bounds[2*d], vol_shape - bounds[2*d+1]),
        keeping particles within the reconstruction volume (with a margin given by bounds)."""
        for di in range(3):
            self.control_points[...,di].data.clamp_(min=self.bounds[2*di],
                                                    max=self.vol_shape[2 - di] - self.bounds[2 * di + 1] - 1e-4)
        return self.control_points

    @torch.no_grad()
    def shake_params(self, shake_width):
        """Perturb control_points in place by uniform noise of the given width, then move any
        particles that ended up outside the pore mask back into it."""
        self.control_points[..., :3] += (torch.rand_like(self.control_points[..., :3]) * 2 - 1) * shake_width
        self.pore_mask.move_coords_to_pore(self.control_points[..., :3])

    @torch.no_grad()
    def calc_distance_matrix(self):
        """Pairwise Euclidean distance between all particles' (single, static) positions."""
        positions = self.control_points.squeeze(1)  # (N, 3)
        return torch.cdist(positions, positions)  # (N, N)


class LinearTrack(StaticTrack):
    """
    Particles moving along a straight line between two control points (start and end), linearly
    interpolated in forward() - the model actually constructed by most of the particle-tracking
    reconstruction scripts under scripts/particle_tracking/ (e.g. via tm.LinearTrack(...)) for
    tracer particles with roughly constant velocity over the scan. displacement_max bounds the
    total start-to-end displacement; bendy_std is accepted but unused (see NurbsTrack, which
    would use it for genuine curvature). PiecewiseLinearTrack generalizes this to more than two
    control points; NurbsTrack fits a smooth curve through them instead of straight segments.
    """
    def __init__(self, dims, device, vol_shape, displacement_max = (0, 0, 0), bendy_std = (0, 0, 0), **kwargs):
        # variations along track
        self.bendy_std = bendy_std
        # total track
        self.displacement_max = torch.tensor(displacement_max, dtype = torch.float32, device = device)
        self.num_control_points = 2
        super().__init__(dims, device, vol_shape, **kwargs)
        
    def forward(self, track_time):
        """Linearly interpolate each particle's position between its start (control point 0)
        and end (control point 1) using track_time as the interpolation fraction in [0, 1]."""
        track_params = self.control_points
        centers_xyz_0, centers_xyz_1 = track_params[:,:1], track_params[:,1:2]  # todo: move_coords_to_pore?
        tracks = centers_xyz_0 + track_time.view(1,-1,1) * (centers_xyz_1 - centers_xyz_0)
        if self.pore_mask is not None:  # confine particles
            tracks = self.pore_mask.move_coords_to_pore(tracks)
        return tracks  # (num_particles, num_sampled_projections, 3)

    def init_close(self, track_params, sigs = None):
        """Re-initialize the start/end control points near track_params' first and last control
        points, perturbed by sigs[:3], then clamp into bounds/displacement limits."""
        centers_xyz = close_control_points(track_params[:, [0,-1],:3], self.pore_mask, sigs[:3])
        self.control_points = nn.Parameter(centers_xyz)
        self.clamp_params()
        return centers_xyz
    
    def _init_tracks(self, num_particles, init_mode = init_random_nearby_vel):
        """Randomly place a start position for each particle (within the pore mask), then pick
        an end position either by a random displacement within displacement_max (init_random) or
        by nudging towards the velocity of nearby existing particles (init_random_nearby_vel,
        useful when adding new particles into an already-reconstructed scene)."""
        centers_xyz = random_coordinates_mask(num_particles, self.pore_mask, self.device)
        if init_mode == init_random:
            centers_xyz2 = close_coordinates_mask(centers_xyz, self.displacement_max, self.pore_mask)
        elif init_mode == init_random_nearby_vel:
            centers_xyz2 = nearby_velocity_init(centers_xyz, self.control_points.data, self.pore_mask)
        else:
            raise NotImplementedError("init_mode not recognized")
        track_params = torch.stack((centers_xyz, centers_xyz2), dim = 1)  # (num_particles, 2, 3)
        self.control_points = nn.Parameter(track_params)
        return self.control_points

    @torch.no_grad()
    def clamp_params(self):
        """Clamp the start-to-end displacement into (-displacement_max, displacement_max) by
        moving the end control point, then defer to StaticTrack.clamp_params to keep every
        control point within the volume bounds."""
        track_params = self.control_points
        displacements_max = self.displacement_max
        displacements = track_params[:, 1, :3] - track_params[:, 0, :3]
        clamped_displacement = torch.clamp(displacements, -displacements_max, displacements_max)
        self.control_points[:, 1, :3] = self.control_points[:,0,:3] + clamped_displacement
        track_params = super().clamp_params()
        return track_params
    
    @property
    def num_control_points(self):
        return self._num_control_points
    
    @num_control_points.setter
    def num_control_points(self, num_control_points):
        self._num_control_points = 2

    @torch.no_grad()
    def calc_distance_matrix(self):
        """Pairwise distance between particles, averaged over both control points (start and
        end), unlike StaticTrack's single-point distance."""
        expanded_a = self.control_points.unsqueeze(1)  # (N, 1, M, 3)
        expanded_b = self.control_points.unsqueeze(0)  # (1, N, M, 3)
        distances = torch.linalg.norm(expanded_a - expanded_b, dim=3)  # (N, N, M)
        mean_dists = distances.mean(dim=2)  # (N, N)
        return mean_dists


class PiecewiseLinearTrack(StaticTrack):
    """
    Generalizes LinearTrack to num_segments straight segments between num_segments + 1 control
    points, for particles whose velocity changes over the course of the scan rather than staying
    constant. NOTE: not currently constructed by any script or test (only mentioned in a comment
    in scripts/particle_tracking/recon_simulated_linear.py as an alternative to LinearTrack) -
    exercise with care if used, since it may not have been run recently.
    """
    def __init__(self, dims, device, vol_shape, num_segments = 2, displacement_max=(0, 0, 0), bendy_std=(0, 0, 0), **kwargs):
        # variations along track
        self.bendy_std = bendy_std
        # total track
        self.displacement_max = torch.tensor(displacement_max, dtype=torch.float32, device=device)
        self.num_control_points = num_segments + 1
        super().__init__(dims, device, vol_shape, **kwargs)

    def forward(self, track_time):
        """Linearly interpolate within whichever of the equal-length segments (between
        consecutive control points) each track_time falls into."""
        track_params = self.control_points
        K, N_vertices, D = track_params.shape
        T = track_time.shape[0]
        N_segments = N_vertices - 1
        delta_t = 1.0 / N_segments

        # Find which segment 'i' each time point belongs to
        segment_idx = torch.floor(track_time / delta_t)
        segment_idx = torch.clamp(segment_idx, max=N_segments - 1).long() #handle t=1 case
        t_start = segment_idx.float() * delta_t
        t_rescaled = (track_time - t_start) / delta_t
        t_rescaled_broadcast = t_rescaled.view(1, T, 1)
        start_points = track_params[:, segment_idx]
        end_points = track_params[:, segment_idx + 1]
        displacement = end_points - start_points
        tracks = start_points + t_rescaled_broadcast * displacement

        if self.pore_mask is not None:  # confine particles
            tracks = self.pore_mask.move_coords_to_pore(tracks)
        return tracks  # (num_particles, num_sampled_projections, 3)

    def init_close(self, track_params, sigs=None):
        """Re-initialize control points near num_control_points positions sampled evenly along
        track_params' own timeline, perturbed by sigs[:3]."""
        timepoints = torch.linspace(0, track_params.shape[1], self.num_control_points, dtype=torch.long, device=self.device)

        centers_xyz = close_control_points(track_params[:, timepoints, :3], self.pore_mask, sigs[:3])
        self.control_points = nn.Parameter(centers_xyz)
        self.clamp_params()
        return centers_xyz


    def _init_tracks(self, num_particles, init_mode = init_random_nearby_vel):
        """Like LinearTrack._init_tracks (random start, then an end position from either a random
        or nearby-velocity displacement), but then lays out num_control_points evenly spaced along
        the straight line between start and end, rather than storing just the two endpoints."""
        centers_xyz = random_coordinates_mask(num_particles, self.pore_mask, self.device)
        if init_mode == init_random:
            centers_xyz_end = close_coordinates_mask(centers_xyz, self.displacement_max, self.pore_mask)
        elif init_mode == init_random_nearby_vel:
            centers_xyz_end = nearby_velocity_init(centers_xyz, self.control_points.data, self.pore_mask)
        else:
            raise NotImplementedError("init_mode not recognized")
        #add linear points along line
        timepoints = torch.linspace(0, 1, self.num_control_points, dtype = torch.float32, device = self.device)
        tracks = centers_xyz.unsqueeze(1) + timepoints.view(1, -1, 1) * (centers_xyz_end - centers_xyz).unsqueeze(1)
        self.control_points = nn.Parameter(tracks)
        return self.control_points

    @torch.no_grad()
    def clamp_params(self):
        """Clamp each segment's displacement into (-displacement_max/num_segments,
        displacement_max/num_segments) in turn (so the total, worst-case end-to-end displacement
        stays within displacement_max), then defer to StaticTrack.clamp_params for volume bounds."""
        track_params = self.control_points
        num_segments = self.num_control_points-1
        displacements_max = self.displacement_max/num_segments
        for i in range(num_segments):
            displacements = track_params[:, i+1, :3] - track_params[:, i, :3]
            clamped_displacement = torch.clamp(displacements, -displacements_max, displacements_max)
            self.control_points[:, i+1, :3] = self.control_points[:, i, :3] + clamped_displacement
        track_params = super().clamp_params()
        return track_params

    @property
    def num_control_points(self):
        return self._num_control_points

    @num_control_points.setter
    def num_control_points(self, num_control_points):
        self._num_control_points = num_control_points

    @torch.no_grad()
    def calc_distance_matrix(self):
        """Pairwise distance between particles, averaged over all num_control_points control points."""
        expanded_a = self.control_points.unsqueeze(1)  # (N, 1, M, 3)
        expanded_b = self.control_points.unsqueeze(0)  # (1, N, M, 3)
        distances = torch.linalg.norm(expanded_a - expanded_b, dim=3)  # (N, N, M)
        mean_dists = distances.mean(dim=2)  # (N, N)
        return mean_dists

class NurbsTrack(LinearTrack):
    """
    Fits a smooth NURBS (Non-Uniform Rational B-Spline) curve of the given degree through
    num_control_points learnable control points (each augmented with a weight, see labels),
    for particles whose real trajectory is curved rather than piecewise-linear. Evaluated via
    De Boor's algorithm (bspline_curve/nurbs_curve/nurbs_track).

    NOTE: the only call site is scripts/particle_tracking/nurbs_demo.py, a standalone
    curve-fitting demo (not a full CT reconstruction) that constructs this class with a
    `sino_params` keyword argument this constructor does not accept (it takes device/vol_shape
    positionally instead) - that call site appears out of sync with this class's current
    constructor. init_close is explicitly marked not updated below, and clamp_params calls
    super().clamp_params(track_params) though LinearTrack.clamp_params takes no such argument -
    treat this class as experimental/not verified to run end-to-end.
    """
    def __init__(self, dims, device, vol_shape, degree, num_control_points, **kwargs):
        self.weight_min = 0.3
        self.weight_mean = 1
        self.weight_std = 0
        self.degree = degree
        self.time_bounds = (0,1)
        self.num_control_points = num_control_points
        self.knot_vector = self.uniform_knot_vector()
        super().__init__(dims, device, vol_shape, **kwargs)
        
    @property
    def num_control_points(self):
        return self._num_control_points

    @num_control_points.setter
    def num_control_points(self, num_control_points):
        self._num_control_points = max(self.degree + 1, num_control_points)
        self.knot_vector = self.uniform_knot_vector()

    @property
    def labels(self):
        return ['center_x', 'center_y', 'center_z', 'weight']
    
    def uniform_knot_vector(self):
        """Build a clamped uniform knot vector (degree-many repeated 0s and 1s around a uniform
        interior) shared by every particle's curve, sized to match num_control_points/degree."""
        # one global and immutable knot vector, not separate for all particles
        knot_vector = torch.cat([torch.zeros(self.degree),
                                 torch.linspace(0, 1, self.num_control_points - self.degree + 1),
                                 torch.ones(self.degree)], dim=0)  # (num_controls + deg + 1)
        return knot_vector
    
    def forward(self, track_time):
        """Evaluate the NURBS curve at each track_time, using track_time itself both as the
        curve parameter u and (rescaled into time_bounds) as the time coordinate passed through
        to nurbs_track."""
        num_tracks = len(self.control_points)
        time_points = (track_time * (self.time_bounds[1] - self.time_bounds[0]) + self.time_bounds[0])
        time_points = time_points.expand(num_tracks, -1)
        tracks = self.nurbs_track(track_time, self.control_points, self.knot_vector, time_points)[...,:self.dims]  # xyz
        if self.pore_mask is not None:  # confine particles
            tracks = self.pore_mask.move_coords_to_pore(tracks)
        return tracks

    def _init_tracks(self, num_particles, init_mode = init_random):
        """Start from a straight-line LinearTrack initialization, then repeat its end point to
        fill out num_control_points control points (so the curve initially coincides with that
        line), perturb them by bendy_std for some initial curvature, and append a weight
        (normal(weight_mean, weight_std)) per control point."""
        random_tracks = super()._init_tracks(num_particles, init_mode)
        # repeat last control point for memory allocation
        random_tracks = torch.cat((random_tracks[:,:-1],) + (random_tracks[:,-1:],) * (self.num_control_points - 1),
                                  dim = -2)
        random_tracks = close_control_points(random_tracks, self.pore_mask, sigs = self.bendy_std)
        weights = torch.normal(mean=self.weight_mean, std = self.weight_std,
                               size=(num_particles, self.num_control_points, 1), device = self.device)
        random_tracks = torch.cat((random_tracks, weights), dim = -1)
        self.control_points = nn.Parameter(random_tracks)
        return self.control_points

    def init_close(self, track_params, sigs = None):
        #TODO not updated
        raise NotImplementedError("Not updated to parameters yet")
        # other track does not need to have the same number of control points
        random_tracks = super().init_close(track_params[:, [0,-1],:3], self.pore_mask, sigs[:3])
        # repeat last control point for memory allocation
        random_tracks = torch.cat((random_tracks[:,:-1],) + (random_tracks[:,-1:],) * (self.num_control_points - 1),
                                  dim = -2)
        random_tracks = close_control_points(random_tracks, self.pore_mask, sigs = (0,0,0))  # straight line
        
        weights = torch.normal(mean=track_params[..., :self.num_control_points, -1:],  # cropping - not exactly the same
                               std = sigs[-1] * torch.ones_like(track_params[..., :self.num_control_points, -1:]))
        close_tracks = torch.cat((random_tracks, weights), dim = -1)
        return close_tracks

    # Convert control points to homogeneous form (x*w, y*w, w)
    @staticmethod
    def to_homogeneous(points):
        dimension = points.shape[-1] - 1  # xyt = 3
        homogeneous_points = torch.stack([points[..., dim] * points[..., -1] for dim in range(dimension)]
                                         + [points[..., -1]], dim=-1)
        return homogeneous_points
    
    # Convert back from homogeneous to Cartesian coordinates (x/w, y/w)
    @staticmethod
    def from_homogeneous(points):
        return points[...,:-1] / points[..., -1:]  # Broadcasting ensures correct division
    
    # De Boor's Algorithm for evaluating a NURBS curve
    def bspline_curve(self, u_values, control_points, knot_vector = None):
        """
        Evaluates the NURBS curve at parameter u using the De Boor algorithm.
        """
        num_tracks, num_control_points, num_dims = control_points.shape
        knot_vector = self.knot_vector if knot_vector is None else knot_vector
        knot_vector = knot_vector.to(u_values.device)
        if u_values.ndim == 1:  # first dimension is track dimension
            u_values = u_values.unsqueeze(0).expand(num_tracks, -1)
            knot_vector = knot_vector.unsqueeze(0).expand(num_tracks, -1)
    
        degree = knot_vector.shape[-1] - num_control_points - 1
        num_tracks, num_samples = u_values.shape
    
        # Find the span indices for all u values
        u_values = u_values.unsqueeze(-1)  # Shape (num_tracks, num_samples, 1 -> num_dims) for broadcasting
        span_indices = (u_values >= knot_vector.unsqueeze(-2)[...,:-1]) & (u_values <= knot_vector.unsqueeze(-2)[...,1:])
        span_indices = span_indices.int().argmax(dim=-1)  # First True index per u_value
    
        # Ensure spans stay within bounds
        k = span_indices = torch.clamp(span_indices, min=degree, max=num_control_points - 1)
    
        # De Boor's recursion formula
        d = torch.stack([control_points.gather(dim = -2, index = j + span_indices.unsqueeze(-1).expand(-1, -1, num_dims) - degree)
                         for j in range(0, degree + 1)], -2)
    
        for r in range(1, degree + 1):
            for j in range(degree, r - 1, -1):
                alpha_indices = span_indices - degree + j
    
                # Compute knot values safely
                # alpha_indices = span_indices - degree + torch.arange(num_tracks, device=u_values.device).view(1, 1, -1)
                knot_left = knot_vector.gather(dim=-1, index=alpha_indices)  # Shape (num_tracks, num_samples)
                knot_right = knot_vector.gather(dim=-1, index=alpha_indices + 1)
    
                # Compute alpha
                alpha_denominators = knot_right - knot_left
                alpha_denominators = torch.where(alpha_denominators == 0,
                                                 torch.ones_like(alpha_denominators), alpha_denominators)  # Prevent div by zero
                alpha = (u_values.squeeze(-1) - knot_left) / alpha_denominators
                alpha = alpha.unsqueeze(-1)  # Shape (num_tracks, num_samples, 1) for broadcasting
    
                # Compute the recursive update
                dc = d.clone()  # Ensure we don't modify in place
                d[..., j, :] = (1 - alpha) * dc[..., j - 1, :] + alpha * dc[..., j, :]
    
        # Select final evaluated points from homogeneous coordinates
        final_points = d[...,degree,:]
    
        return final_points

    @torch.no_grad()
    def clamp_params(self, track_params):
        """Clamp control-point weights to stay at or above weight_min (weights must be positive
        for the homogeneous NURBS representation to be well-defined), on top of whatever the
        parent clamp_params does for the xyz coordinates."""
        super().clamp_params(track_params)
        track_params[:,:,-1] = torch.clamp(track_params[:,:,-1], min = self.weight_min)  # positive weights
        return track_params

    def calc_distance_matrix(self):
        raise NotImplementedError("Only get average distance of control points, not track necessarily")
        # super().calc_distance_matrix(track_params[...,:3])


    def nurbs_curve(self, u_values, control_points, knot_vector = None):
        """Evaluate the (rational) NURBS curve at u_values: converts control_points to
        homogeneous coordinates, runs the plain B-spline evaluation (bspline_curve), then
        converts back - this is what makes the curve rational (weighted) rather than a plain
        B-spline."""
        knot_vector = self.knot_vector if knot_vector is None else knot_vector
        # Convert to Homogeneous coordinates
        degree = knot_vector.shape[-1] - control_points.shape[1] - 1
        homogeneous_points = self.to_homogeneous(control_points)
        final_points = self.bspline_curve(u_values, homogeneous_points, knot_vector)
        # Convert back to Cartesian coordinates
        final_points = self.from_homogeneous(final_points)
        return final_points
    
    def nurbs_track(self, u_values, control_points, knot_vector = None, time_points = None):
        """
        Evaluates the NURBS curve at parameter u using the De Boor algorithm.
        """
        knot_vector = self.knot_vector if knot_vector is None else knot_vector
        degree = knot_vector.shape[-1] - control_points.shape[1] - 1
        curve_points = self.nurbs_curve(u_values, control_points[..., list(range(self.dims)) + [-1]], knot_vector)  # x y w no time dependence
        # time_points = evaluate_bspline(u_values, control_points[...,degree - 2:-degree + 1,-2:-1],
        #                                knot_vector [...,degree - 1:-degree + 1], 1)  # piecewise linear
        if time_points is None:
            time_points = (control_points[:,:1,-2] + u_values * control_points[:,-1:,-2])  # only start and end matter
        points = torch.concatenate([curve_points, time_points.unsqueeze(-1)], -1)
        return points

    @staticmethod
    def find_u_from_t(batch_t_values, time_points, u_values_full):
        """Invert the (assumed linear, start/end-only) time parametrization to recover, for each
        requested time in batch_t_values, the curve parameter u and its index into u_values_full
        (used by nurbs_demo.py to look up ground-truth samples at specific times)."""
        t0, t1 = time_points[...,:1], time_points[...,-1:]
        # batch_t_values = t0 + u * (t1 - t0) 
        u = (batch_t_values - t0) / (t1 - t0) * (1-0)
    
        # soften the blow
        # u = torch.clamp(u, 0., 1.)  # todo: relax value
        u_indices = torch.sum(u.unsqueeze(-1) > u_values_full.unsqueeze(-2), dim = -1).detach().numpy()  # temp shape (num_tracks, num_samples, num_projections)
        return u, u_indices


