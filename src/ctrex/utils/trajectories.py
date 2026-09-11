#!/usr/bin/python
"""!
Created on Wed Jun 25 10:18:35 2014

@author wannesg
"""
import torch
import math
from torch import nn
from itertools import chain

from ctrex.utils.datasets import Detector
from ctrex.optimization import torchtools


class Volume:  # Data shape descriptor based on the CT geometry, no data here
    """Describes the shape and centre of the reconstruction volume (depth/height/width in
    voxels) that a Trajectory derives from its detector and scan geometry. Holds no actual
    voxel data itself."""

    def __init__(self, depth, height, width):
        self.depth, self.height, self.width = depth, height, width
        self.vol_centre = None

    @property
    def vol_shape(self):
        return self.depth, self.height, self.width

    @property
    def shape(self):
        return self.depth, self.height, self.width

    @property
    def max_side(self):
        return max(self.width, self.height)
    
    def set_vol_centre(self, device):
        """Compute and cache the volume's geometric centre (in xyz voxel order) on `device`.
        Called once at setup time; the comment below is the reason why."""
        # set once, don't alter during optimization to avoid breaking torch computation graph
        self.vol_centre = torch.tensor(self.shape[::-1], device = device) / 2
        return self.vol_centre


class Trajectory(torchtools.OptimModule):
    """General helical cone-beam scan trajectory: holds the learnable/fixed geometry
    parameters (centre of rotation, detector tilt/skew, scan angles, helical pitch, source/
    detector distances, pixel size, etc.), derives the matching reconstruction Volume, and
    converts between voxel coordinates and detector projections (`project_voxels`/`get_rays`)
    for a stack of projections. Circular, screw and other simpler trajectories are just this
    class with some parameters left at their defaults (e.g. zero helical pitch).
    """
    # store params for a general helical trajectory. Circle, Screw and rotate are derived
    # Here different types of geometries can be made in a clear way, without messing in detector
    # Track detector and volume to derive defaults depending on detector or volume shape
    
    def __init__(self, detector: Detector, device ='cpu', requires_grad = True, extend_fov = True):
        """Declare every trajectory parameter/buffer (most default to a simple circular scan:
        zero skew/tilt/helical pitch, 0-360 degree sweep) and derive the reconstruction volume
        from `detector` (see `_setup_volume` and the `extend_fov` note there)."""
        super().__init__()
        self.detector = detector  # type: Detector
        self.device = device
        param_kwargs = {'dtype':torch.float32, 'device':device, 'requires_grad': requires_grad}
        kwargs = {'dtype':torch.float32, 'device':device}
        self.centre_of_rotation = nn.Parameter(torch.tensor(self.detector.width / 2, **param_kwargs))
        self.skew = nn.Parameter(torch.tensor(0, **param_kwargs))
        self.tilt = nn.Parameter(torch.tensor(0, **param_kwargs))
        self.first_angle = nn.Parameter(torch.tensor(0, **param_kwargs))
        self.last_angle = nn.Parameter(torch.tensor(360, **param_kwargs))
        self.angle_drift = nn.Parameter(torch.tensor([0] * max(0, self.num_projections - 2), **param_kwargs))
        self.register_buffer('clockwise', torch.tensor(1, dtype = torch.int32, device = device))
        self.register_buffer('sod',torch.tensor(300, **kwargs))
        self.register_buffer('sdd',torch.tensor(1000, **kwargs))
        self.vertical_centre = nn.Parameter(torch.tensor(self.detector.height / 2, **param_kwargs))
        self.horizontal_centre = nn.Parameter(torch.tensor(self.detector.width / 2, **param_kwargs))
        self.register_buffer('pixel_size', torch.tensor(0.1, **kwargs))
        self.register_buffer('translation_axis', torch.tensor([0,0,1], **kwargs))
        self.register_buffer('rotation_axis', torch.tensor([0,0,1], **kwargs))
        self.helical_pitch = nn.Parameter(torch.tensor(0, **param_kwargs))
        self.register_buffer('signs', torch.tensor([1.,-1.,-1], **kwargs))
        self.register_buffer('subviews', torch.tensor(1, dtype = torch.uint32, device = self.device))




        """if True (default), the reconstruction volume is widened/re-centred around
            centre_of_rotation to cover the full field of view available from an offset-detector
            (half-fan) scan geometry - correct for reconstruction paths that derive their own vol_shape
            from trajectory.volume (e.g. the static-matrix/array path, see ctrex.utils.templates).
            Set False to instead keep the volume fixed at exactly (detector.height, detector.width,
            detector.width), centred on (height/2, width/2, width/2) regardless of centre_of_rotation -
            required whenever some OTHER part of the pipeline independently assumes that grid (e.g.
            particle tracking's pore mask and track-model vol_shape, which are sized to detector.width
            in the calling script rather than read off trajectory.volume). Leaving this mismatched is a
            silent bug: voxel indexing desyncs between the reprojection geometry and anything that
            floors/clamps particle positions against such a fixed-size mask, corrupting which particles
            are judged to be inside the pore vs. in the grain whenever centre_of_rotation drifts off
            detector.width/2, even by a fraction of a voxel."""
        self.extend_fov = extend_fov
        self._setup_volume()
        self.learning_rate = 0  # actively set learning rates for desired parameters

    def get_params(self, detach=True):
        """Return dict with both trainable and fixed geometry values."""
        params = {}

        # Collect learnable parameters
        for name, p in self.named_parameters(recurse=True):
            params[name] = p.detach().cpu().item() if detach else p

        # Collect fixed buffers
        for name, b in self.named_buffers(recurse=True):
            params[name] = b.detach().cpu().item() if detach else b

        return params

    def set_params(self, params):
        """Write `params` (as produced by `get_params`, or `sino_params`) back onto this
        trajectory's own parameters/buffers by name - trying both the exact name and a
        variant with underscores replaced by spaces. Anything left over after that is set as
        a plain attribute instead. Re-derives the volume afterwards since geometry may have
        changed."""
        params = params.copy()
        for name, p in chain(self.named_parameters(recurse=True),
                             self.named_buffers(recurse=True)):
            for name_variant in (name, name.replace('_',' ')):
                if name_variant in params:
                    with torch.no_grad():
                        try:
                            p.copy_(torch.as_tensor(params.pop(name_variant), dtype=p.dtype, device=p.device))
                        except RuntimeError:
                            print(f'cannot set {name_variant}')
        for key, val in params.items():
            self.__setattr__(key, val)
        self._setup_volume()

    @property
    def num_projections(self):
        """Number of projections, read from the detector."""
        return self.detector.num_projections

    @num_projections.setter
    def num_projections(self, num_projections):
        """Update the detector's projection count and resize `angle_drift` to match (drift is
        only defined for the projections strictly between the first and last angle)."""
        self.detector.num_projections = num_projections
        self.angle_drift = nn.Parameter(torch.tensor([0] * max(0, self.num_projections - 2),
                                                     dtype = torch.float32,
                                                     device = self.angle_drift.device,
                                                     requires_grad = self.angle_drift.requires_grad))

    @property
    def angles(self):
        """Per-projection scan angle: an evenly spaced sweep from `first_angle` to
        `last_angle`, perturbed by the learnable `angle_drift` on every projection except
        the first and last (which anchor the sweep)."""
        angles = torch.linspace(self.first_angle, self.last_angle, max(1, self.num_projections),
                                device=self.angle_drift.device, dtype=torch.float32)
        angles[1:-1] = angles[1:-1] + self.angle_drift
        return angles.to(self.device)
    
    @angles.setter
    def angles(self, angles: torch.Tensor):
        """Set an explicit angle array: derives `num_projections`, `first_angle`,
        `last_angle`, and backs out the `angle_drift` needed to reproduce `angles` exactly."""
        angles = torch.as_tensor(angles, dtype = torch.float32, device = self.device)
        with torch.no_grad():
            self.num_projections = angles.shape[0]
            self.first_angle.copy_(angles[0])
            self.last_angle.copy_(angles[-1])
            default_angles = torch.linspace(self.first_angle, self.last_angle, max(1, self.num_projections),
                                            device=self.device, dtype=torch.float32)
            self.angle_drift.copy_(angles[1:-1] - default_angles[1:-1])

    @property
    def num_rotations(self):
        """Total angular sweep expressed in number of full rotations."""
        return (self.last_angle - self.first_angle) / 360

    @num_rotations.setter
    def num_rotations(self, num_rotations):
        """Set the sweep by moving `last_angle` to give `num_rotations` full turns starting
        from the current `first_angle`."""
        with torch.no_grad():
            self.last_angle.copy_(num_rotations * 360 + self.first_angle)
        pass

    @property
    def proj_per_rot(self):
        """Average number of projections per full rotation."""
        return (self.num_projections - 1) / self.num_rotations

    @proj_per_rot.setter
    def proj_per_rot(self, proj_per_rot):
        """Set `num_rotations` so that the current projection count yields `proj_per_rot`
        projections per rotation."""
        self.num_rotations = (self.num_projections - 1) / proj_per_rot

    def projection_indices(self, sampled_projections):
        """Resolve `sampled_projections` (an index array or slice) into concrete projection
        index tensors on this trajectory's device."""
        projections = torch.arange(0, self.num_projections, dtype = torch.int32, device = self.device)
        sampled_projections = projections[sampled_projections].to(self.device)
        return sampled_projections

    def projection_time(self, sampled_projections):
        """Normalized acquisition time (0 to 1 across the scan) for `sampled_projections`,
        used to evaluate time-dependent track/shape models at the right instant."""
        # assume constant time difference between projection readout
        track_time = torch.linspace(0, 1, self.num_projections, device = self.device)[sampled_projections]
        return track_time

    def _setup_volume(self):
        """Derive the reconstruction Volume's depth/width from the detector, helical pitch and
        (depending on `extend_fov`, see the note in `__init__`) the centre of rotation. Called
        once at setup time and again whenever `set_params` changes the geometry."""
        # bounding box to define voxel coordinates, with the default voxel size
        # this is done once the initial geometry parameters are set, not adjusted during optimization
        if self.extend_fov:
            cor = self.centre_of_rotation.item()
            width = int(2 * abs(max(cor - 0, self.detector.width - cor)))
        else:
            width = self.detector.width
        depth = self.detector.height + int(self.helical_pitch * self.num_rotations / self.voxel_size / 2)

        self.volume = Volume(depth, width, width)  # z,y,x
        self.volume.set_vol_centre(self.device)

    @property
    def voxel_size(self):
        """Physical size (mm) of one reconstruction voxel at the centre of rotation, derived
        from the detector pixel size and the source/detector/rotation-axis distances (i.e.
        the magnification at the rotation axis)."""
        half_opening_angle = torch.atan(self.pixel_size * self.detector.width * .5 / self.sdd)
        voxel_size = (abs(self.pixel_size) * (self.sod/self.sdd) *
                      torch.cos(half_opening_angle))
        return voxel_size

    def roi_to_voi(self, roi = None):
        """Translate a detector region of interest `roi` (defaults to the full detector) into
        the matching volume of interest, padding vertically for helical scans and widening
        around `centre_of_rotation` for off-centre (half-fan) geometries.

        Returns:
            voi: (depth, height, width) slice tuple in the volume's voxel coordinates.
            voi_shape: shape of that voi.
            voi_centre: xyz centre of that voi, in voxel coordinates.
        """
        # for a detector roi, define a suitable voi in the volume domain
        # helical trajectories can extend the voi
        vertical_pad = self.volume.depth - self.detector.height
        # for off_centre rotation axes, the width increases
        cor = self.centre_of_rotation
        if roi is None:
            roi = (slice(0, self.detector.height), slice(0, self.detector.width))
        extended_width = 2 * abs(max(cor - roi[1].start , roi[1].stop - cor))
        voi = (slice(roi[0].start - 0, roi[0].stop + vertical_pad),
               slice(int(self.volume.width - extended_width) / 2, int(self.volume.width + extended_width) / 2),
               slice(int(self.volume.width - extended_width) / 2, int(self.volume.width + extended_width) / 2))
        voi_shape = (voi[0].stop - voi[0].start,
                     voi[1].stop - voi[1].start,
                     voi[2].stop - voi[2].start)  # zyx
        voi_centre = torch.tensor([(voi[2].stop + voi[2].start) / 2,
                                   (voi[1].stop + voi[1].start) / 2,
                                   (voi[0].stop + voi[0].start) / 2], device = self.device)  # xyz
        return voi, voi_shape, voi_centre

    def voi_to_roi(self, voi):
        """NOTE: not currently used/called anywhere in src/ or scripts/. Inverse of
        `roi_to_voi`: projects the 8 corners of a volume of interest `voi` through every
        projection angle and returns the detector region that would need to be loaded to
        reconstruct it. (The v-slice below appears to reuse the same min/max as the u-slice -
        worth a second look if this is ever revived.)"""
        # for a volume of interest, calculate what roi you need to load on the detector to reconstruct this.
        d0, d1 = voi[0].start, voi[0].stop
        h0, h1 = voi[1].start, voi[1].stop
        w0, w1 = voi[2].start, voi[2].stop
        corner_points = torch.tensor([[w0, h0, d0],
                                      [w0, h0, d1],
                                      [w0, h1, d0],
                                      [w1, h1, d1],
                                      [w1, h0, d0],
                                      [w1, h1, d1],
                                      [w1, h1, d0]], device = self.device)
        views = self.calc_trajectory(slice(0, self.num_projections))
        projected_corner_points = self.project_voxels(corner_points, views)
        roi = (slice(max(0, int(projected_corner_points[1][:].min().floor())),
                     min(self.detector.height, int(projected_corner_points[1][:].max().ceil()))),  # v
               slice(max(0, int(projected_corner_points[1][:].min().floor())),
                     min(self.detector.width, int(projected_corner_points[1][:].max().ceil()))))  # u
        return roi

    @property
    def roi_vol_shape(self) -> tuple[int, int, int]:
        """NOTE: not currently used/called anywhere in src/ or scripts/. Volume shape matching
        the detector's own roi height, at the full detector width."""
        # look for the window around the centre of rotation
        # cor = self.centre_of_rotation
        # rwidth = abs(max(cor - self.detector.roi[1].start, self.detector.roi[1].stop - cor))
        return self.detector.rheight, self.detector.width, self.detector.width

    @property
    def roi_vol(self):
        """NOTE: not currently used/called anywhere in src/ or scripts/. Volume of interest
        slice tuple matching the detector's own roi (vertically), at the full detector width."""
        roi_vol = (self.detector.roi[0], slice(0, self.detector.width), slice(0, self.detector.width))
        return roi_vol

    @property
    def sino_params(self):
        """Full geometry parameter dict for this trajectory: every learnable parameter and
        fixed buffer plus detector shape/roi/dimension, with `last_angle` dropped in favour of
        `proj_per_rot` (which lets `num_projections` vary while keeping the same angular
        spacing). Used to hand trajectory state to `CTScan`/`CTSimulation` and to round-trip
        through `set_params`."""
        sino_params = dict(self.named_parameters())
        sino_params.update(**dict(self.named_buffers()))
        sino_params['height'] = self.detector.height
        sino_params['width'] = self.detector.width
        sino_params['num_projections'] = self.num_projections
        sino_params['proj_per_rot'] = self.proj_per_rot
        sino_params.pop('last_angle')  # given by proj_per_rot, to vary num_projections
        sino_params['roi'] = self.detector.roi
        sino_params['dimension'] = self.detector.dimension
        return sino_params

    def detector_trajectory(self, sampled_projections):
        """! Computes the trajectory for the detector. """
        xoff = (self.detector.width / 2 - self.horizontal_centre) * abs(self.pixel_size)
        zoff = (self.detector.height / 2 - self.vertical_centre) * abs(self.pixel_size)
        sdd = torch.as_tensor(self.sdd, device = self.device)
        detector_trl = torch.stack((xoff, sdd, -zoff)).unsqueeze(0)
        detector_rot = torch.stack((0 * self.skew, self.skew, self.tilt)).unsqueeze(0)
        det_traj = euler_to_matrix(torch.deg2rad(detector_rot), axes='szxy')
        det_traj[..., :3, 3] = detector_trl  # detector center
        det_traj = det_traj.expand(len(sampled_projections), int(self.subviews), 4, 4)
        return det_traj
        
    def source_trajectory(self, sampled_projections):
        """Source position (homogeneous xyz1, fixed at the origin) broadcast over
        `sampled_projections` and the subviews-per-projection."""
        # fixed source position
        src_traj = torch.tensor([0,0,0,1], dtype=torch.float32, device = self.device)  # xyz1
        src_traj = src_traj.expand(len(sampled_projections), int(self.subviews), 4)
        return src_traj

    def volume_trajectory(self, sampled_projections):
        """Rotation + translation of the volume for each of `sampled_projections` (and each
        intermediate sub-angle within a projection, to simulate a smooth/fly scan): rotates
        around `rotation_axis` by the (sub-angle-adjusted) scan angle, and translates the
        volume centre off the rotation axis by `centre_of_rotation` plus the helical
        (`helical_pitch`) travel at that point in time."""
        # rotation part
        # calculate intermediate angles to simulate smooth scan / fly scan
        subviews = int(self.subviews)
        sub_angles = torch.linspace(-0.5 + 0.5 / subviews, 0.5 - 0.5 / subviews, subviews, device=self.device)
        sub_angles *= self.angles[1] - self.angles[0]  # angular spacing
        all_angles = self.angles[sampled_projections].unsqueeze(1) + sub_angles.unsqueeze(0)
        all_angles = torch.deg2rad(all_angles)
        trajvol = rotation_matrix(self.clockwise * all_angles,
                                  self.rotation_axis.expand(all_angles.shape + (3,)))

        # translation part describing centre of volume
        rotpos = (self.centre_of_rotation - self.horizontal_centre) * self.pixel_size * self.sod / self.sdd
        sod = torch.as_tensor(self.sod, device = self.device)
        axispos = torch.stack((rotpos, sod, 0 * sod), dim=-1).view(-1, 3)
        centers = (axispos
                   + self.helical_pitch * self.translation_axis.view(-1, 3) *
                   (self.projection_time(sampled_projections).unsqueeze(-1) - 0.5)  # - 0.5: center passes in middle of scan
                   * self.num_rotations)  # helical pitch = mm travel per rotation
        trajvol[..., :3, 3] = centers.unsqueeze(1)  # (num_projections, num_subviews, 4, 4)
        return trajvol

    def calc_trajectory(self, sampled_projections):
        """! Set up the detector, source, volume trajectory and make a view stack"""
        assert self.num_projections != 0, 'no images'
        try:
            sampled_projections = torch.arange(0, self.num_projections, device = self.device)[sampled_projections]
        except IndexError:  # somehow [0,1,2] slices differently than [tensor(0), tensor(1), tensor(2)]?
            sampled_projections = torch.arange(0, self.num_projections, device = self.device)[torch.as_tensor(sampled_projections)]

        src_traj = self.source_trajectory(sampled_projections)
        vol_traj = self.volume_trajectory(sampled_projections)
        det_traj = self.detector_trajectory(sampled_projections)
    
        # transform for volume rotation like euler_matrix(-ry_v,-rx_v,-rz_v, axes='ryxz')
        volume_center = torch.clone(vol_traj[...,3:4])  # in homogeneous coordinates xyz1
        # volume_transform = vol_traj
        vol_traj[..., :3, 3] = 0  # no translation here
    
        # append volume translation to rotation part
        volume_center = torch.matmul(vol_traj, volume_center)
        vol_traj = torch.cat((vol_traj[..., :3],
                              torch.tensor([-1,-1,-1,1], device = self.device).unsqueeze(-1) * volume_center),
                             dim=-1)  # inverse translation
    
        # transform source position to volume coordinates
        source_center = torch.einsum('...ij,...j->...i', vol_traj, src_traj)
    
        # transform detector pose to volume coordinates
        detector_transform = torch.matmul(vol_traj, det_traj)

        # subtract source position from detector position
        translate = detector_transform[..., :3, 3]
        detector_transform[..., :3, 3] = (translate - source_center[..., :3])

        # combine into a stack of views
        view_stack = torch.cat((detector_transform[...,:3,:4], source_center.unsqueeze(-2)), dim = -2)
        return view_stack  # (num_projections, subviews, 4, 4)

    def forward(self, sampled_projections):
        """nn.Module entry point, simply delegates to `calc_trajectory`."""
        return self.calc_trajectory(sampled_projections)

    def project_voxels(self, voxels, views):
        """Project `voxels` (voxel coordinates) onto the detector for each view in `views`
        (as produced by `calc_trajectory`), by intersecting the ray from source through each
        voxel with the detector plane.

        Returns:
            u_projected: horizontal detector pixel coordinate per voxel per projection,
                adjusted for the detector roi.
            v_projected: vertical detector pixel coordinate, likewise roi-adjusted.
            mag_particle: magnification factor (detector-plane distance over source-to-voxel
                distance) at each voxel/projection.
        """
        # vectors are generally in the volume coordinate space with unit mm, centred on the rotation axis.
        # For reconstruction, the volume grid coordinates are static and the source and detector virtually revolve around the sample
        u_axes = views[:, 0, :3, 0]  # x  (num_projections, 3)
        v_axes = views[:, 0, :3, 2]  # z  (num_projections, 3)
        detector_centers = views[:, 0, :3, 3]  # w (num_projections, 3)
        detector_normals = torch.linalg.cross(u_axes, v_axes, dim = -1)  # (num_projections, 3)
        detector_normals = detector_normals / torch.linalg.norm(detector_normals, axis=-1, keepdims=True)  # u-axes and v_axes are orthonormal, so not needed

        # create orthogonalized axes tus and tvs: dot with u_axes and v_axes and see what happens
        uus = torch.einsum('...ij,...ij->...i', u_axes, u_axes).unsqueeze(-1)  # (num_projections,1)
        vvs = torch.einsum('...ij,...ij->...i', v_axes, v_axes).unsqueeze(-1)  # (num_projections,1)
        uvs = torch.einsum('...ij,...ij->...i', u_axes, v_axes).unsqueeze(-1)  # (num_projections,1)
        tu = ((vvs * u_axes - uvs * v_axes) / (uus * vvs - uvs * uvs)).unsqueeze(0)
        tv = ((uus * v_axes - uvs * u_axes) / (uus * vvs - uvs * uvs)).unsqueeze(0)
        
        # change unit of length to mm and transform coordinate system to project points onto detector
        points = self.voxels_to_world(voxels)  # (num_centers, num_projections, 3)
        source_points = views[:, 0, 3, :3].unsqueeze(0)  # (1, num_projections, 3)
        projected_sourcedetector_distance = torch.einsum('...ij,...ij->...i', detector_centers, detector_normals).unsqueeze(-1)
        projected_source_point_distance = torch.einsum('...ij,...ij->...i', points - source_points, detector_normals).unsqueeze(-1)  # dot for each individual projection
        projected_point = (points - source_points) / projected_source_point_distance * projected_sourcedetector_distance - detector_centers
        u_projected = 0.5 * self.detector.width + torch.einsum('...ij,...ij->...i', projected_point, tu) / self.pixel_size
        v_projected = 0.5 * self.detector.height - torch.einsum('...ij,...ij->...i', projected_point, tv) / self.pixel_size
        mag_particle = ((torch.linalg.norm(projected_point + detector_centers, axis = -1)
                         / torch.linalg.norm(points - source_points, axis = -1)))  # TODO slowest in backwards
        
        # adjust for roi
        u_projected = u_projected - self.detector.roi[1].start
        v_projected = v_projected - self.detector.roi[0].start
        
        # some sanity checks
        # expected_magnification = trajectory.sdd / trajectory.sod
        # average_magnification = 1 / (1 / mag_particle).mean(axis = 1)  # sod is in denominator, so invert first
        # relative orientation of the particles w.r.t. the optical axis can be calculated here, but they're symmetric
        return u_projected, v_projected, mag_particle
    
    def get_rays(self, u_grid, v_grid, views):  # like prepareRayTracer in CTrex
        """Inverse of `project_voxels`: for detector pixel grids `u_grid`/`v_grid` and the
        given `views`, return the world-space ray origin (source position) and direction
        through each pixel, for ray-tracing style projectors."""
        origins = views[:, 0, 3, :3].unsqueeze(0)  # (1, num_projections, 3) (no subviews)
        
        # convert to mm coordinates with vertical axis pointing up, adjusting for roi
        u = self.pixel_size * ((u_grid + self.detector.roi[1].start) - self.detector.width / 2)
        v = self.pixel_size * ((v_grid + self.detector.roi[0].start) - self.detector.height / 2)
        
        detector_transform = views[:, 0, :3, :].unsqueeze(0).unsqueeze(-3).unsqueeze(-3)  # (1, num_projections, 1, 1 3, 4)
        pixel_coordinates = torch.stack([u, u*0, v, u*0+1], dim = -1)  # homogeneous coordinates in detector frame
        pixel_coordinates[...,:3] = pixel_coordinates[...,:3] * self.signs
        directions = torch.einsum('...ij,...j->...i', detector_transform, pixel_coordinates)  # (1, num_projections, n_u, n_v 3)
        return origins, directions
    
    def voxels_to_world(self, voxel_coordinates_xyz):
        """Convert voxel coordinates to world-space mm coordinates, centred on the rotation
        axis (inverse of `world_to_voxels`)."""
        world_coordinates = self.signs * self.voxel_size * (voxel_coordinates_xyz - self.volume.vol_centre)
        return world_coordinates  # in units of mm, with zero at center of rotation
    
    def world_to_voxels(self, world_coordinates_xyz):
        """Convert world-space mm coordinates back to voxel coordinates (inverse of
        `voxels_to_world`)."""
        voxel_coordinates = self.signs * world_coordinates_xyz / float(self.voxel_size) + self.volume.vol_centre
        return voxel_coordinates

    def distances_entry_exit(self, volume, centers_time):
        """NOTE: not currently called anywhere in src/ or scripts/ (only referenced from a
        commented-out line in ctrex/projectors/static_projectors.py). Estimates the
        min/max distance (in voxel units) from the source at which a ray could enter/exit
        `volume`'s bounding box, with a margin for the cone-beam angle - meant for bounding a
        ray-marching/ray-tracing search range."""
        depth, height, width = volume.shape
        half_diagonal = math.sqrt(width ** 2 + height ** 2) / 2 * 1.2  # add some margin for cone beam
        min_distance = float(self.sod / self.voxel_size - half_diagonal)
        max_distance = float(self.sod / self.voxel_size + half_diagonal)
        return min_distance, max_distance


class MultiTrajectory:
    """NOTE: not currently used/called anywhere in src/ or scripts/, and its constructor call
    below looks broken as it stands: it passes `volume` positionally into `Trajectory`'s
    `device` parameter slot (`Trajectory.__init__` takes `(detector, device=..., requires_grad=...,
    extend_fov=...)`, with no `volume` parameter at all). Intended, presumably, to hold several
    Trajectory instances (e.g. for a multi-source/multi-detector setup) - would need fixing
    before use."""

    def __init__(self, detector: Detector, volume: Volume, device ='cpu', num_traj = 2):
        self.trajectories = [Trajectory(detector, volume, device) for _ in range(num_traj)]


# map axes strings to/from tuples of inner axis, parity, repetition, frame
_axes2tuple = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_tuple2axes = dict((v, k) for k, v in _axes2tuple.items())


def euler_to_matrix(angles, axes='sxyz', derive_axis =''):
    """!
    @param angles: torch.tensor: optional + (3,)
    @param axes: One of 24 axis sequences as string or encoded tuple
    @param derive_axis: derive some axis ijk
    @return transformation matrices in shape angles.shape(:-1) + (4,4)
    """

    try:
        firstaxis, parity, repetition, frame = _axes2tuple[axes]
    except (AttributeError, KeyError):
        _ = _tuple2axes[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    angle_i, angle_j, angle_k = torch.moveaxis(angles, source=-1, destination=0)

    # axis sequences for Euler angles
    _next_axis = [1, 2, 0, 1]
    i = firstaxis
    j = _next_axis[i + parity]
    k = _next_axis[i - parity + 1]

    if frame:
        angle_i, angle_k = angle_k, angle_i
    if parity:
        angle_i, angle_j, angle_k = -angle_i, -angle_j, -angle_k

    di, dj, dk = 'i' in derive_axis, 'j' in derive_axis, 'k' in derive_axis
    ndi, ndj, ndk = not di, not dj, not dk

    sin_i, cos_i = (torch.cos(angle_i), - torch.sin(angle_i)) if di else (torch.sin(angle_i), torch.cos(angle_i))
    sin_j, cos_j = (torch.cos(angle_j), - torch.sin(angle_j)) if dj else (torch.sin(angle_j), torch.cos(angle_j))
    sin_k, cos_k = (torch.cos(angle_k), - torch.sin(angle_k)) if dk else (torch.sin(angle_k), torch.cos(angle_k))
    cos_i_cos_k, cos_i_sin_k = cos_i * cos_k, cos_i * sin_k
    sin_i_cos_k, sin_i_sin_k = sin_i * cos_k, sin_i * sin_k

    mat = torch.eye(4, dtype=torch.float32, device = angles.device).tile(angles.shape[:-1] + (1,1))
    if repetition:
        mat[..., i, i] = cos_j * ndi * ndk
        mat[..., i, j] = sin_j * sin_i * ndk
        mat[..., i, k] = sin_j * cos_i * ndk
        mat[..., j, i] = sin_j * sin_k * ndi
        mat[..., j, j] = -cos_j * sin_i_sin_k + cos_i_cos_k * ndj
        mat[..., j, k] = -cos_j * cos_i_sin_k - sin_i_cos_k * ndj
        mat[..., k, i] = -sin_j * cos_k * ndi
        mat[..., k, j] = cos_j * sin_i_cos_k + cos_i_sin_k * ndj
        mat[..., k, k] = cos_j * cos_i_cos_k - sin_i_sin_k * ndj
    else:
        mat[..., i, i] = cos_j * cos_k * ndi
        mat[..., i, j] = sin_j * sin_i_cos_k - cos_i_sin_k * ndj
        mat[..., i, k] = sin_j * cos_i_cos_k + sin_i_sin_k * ndj
        mat[..., j, i] = cos_j * sin_k * ndi
        mat[..., j, j] = sin_j * sin_i_sin_k + cos_i_cos_k * ndj
        mat[..., j, k] = sin_j * cos_i_sin_k - sin_i_cos_k * ndj
        mat[..., k, i] = -sin_j * ndi * ndk
        mat[..., k, j] = cos_j * sin_i * ndk
        mat[..., k, k] = cos_j * cos_i * ndk
    return mat


def rotation_matrix(angle, direction, point=None):
    """!
    Return matrix to rotate about axis defined by point and direction.
    
    Supports a stack of angles, directions and points
    
    Copied and adapted from transformations.py

    # >>> from transformations import rotation_matrix
    >>> angles = torch.tensor([math.pi/3, -math.pi/2])
    >>> dir_axis = torch.tensor([[1,1,0]]*2)
    >>> points = torch.ones((2,3))
    >>> rotation_matrix(angles[0:2], dir_axis[0:2], points)
    array([[[ 7.50000000e-01,  2.50000000e-01,  6.12372436e-01,
             -6.12372436e-01],
            [ 2.50000000e-01,  7.50000000e-01, -6.12372436e-01,
              6.12372436e-01],
            [-6.12372436e-01,  6.12372436e-01,  5.00000000e-01,
              5.00000000e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              1.00000000e+00]],
    <BLANKLINE>
           [[ 5.00000000e-01,  5.00000000e-01, -7.07106781e-01,
              7.07106781e-01],
            [ 5.00000000e-01,  5.00000000e-01,  7.07106781e-01,
             -7.07106781e-01],
            [ 7.07106781e-01, -7.07106781e-01,  6.12323400e-17,
              1.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              1.00000000e+00]]])
    >>> rotation_matrix(angles[0], dir_axis[0], points[0])
    array([[ 0.75      ,  0.25      ,  0.61237244, -0.61237244],
           [ 0.25      ,  0.75      , -0.61237244,  0.61237244],
           [-0.61237244,  0.61237244,  0.5       ,  0.5       ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    
    """
    sina = torch.sin(angle).unsqueeze(-1)
    cosa = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
    # rotation matrix around unit vector

    direction = direction[..., :3] / torch.norm(direction[..., :3], dim=-1, keepdim = True)
    outer = torch.einsum('...i,...j->...ij', direction, direction)  # outer product without side effects

    direction = direction * sina
    d0, d1, d2 = direction[...,0], direction[...,1], direction[...,2]
    dirmat = torch.stack((torch.stack((0 * d0, -d2, d1), dim = -1),
                          torch.stack((d2, 0 * d0, -d0), dim = -1),
                          torch.stack((-d1, d0, 0 * d0), dim = -1)), dim = -2)
    rot = (torch.eye(3, device = angle.device).tile(angle.shape + (1,1)) * cosa
           + outer * (1.0 - cosa)
           + dirmat)  # (..., 3, 3)

    if point is not None:
        # rotation not around origin
        point = point[...,:3].to(torch.float32)
        trl = point - torch.einsum('...ij,...j->...i', rot, point)
    else:
        trl = rot[...,-1] * 0
    mat = torch.cat((rot, trl.unsqueeze(-1)), dim = -1)  # (..., 3, 4)
    mat = torch.cat((mat, torch.tensor([0,0,0,1], device = mat.device).tile(angle.shape + (1, 1))),
                    dim = -2)  # (..., 4, 4)
    return mat


def project_coordinates_parallel(sino_params, centers_x, centers_y, centers_z):
    """NOTE: not currently called anywhere in src/ or scripts/. Simplified, parallel-beam
    variant of `Trajectory.project_voxels`/`get_rays`: projects xyz centers onto detector (u,v)
    coordinates assuming a parallel-beam geometry (no magnification, no cone/tilt/skew), using
    just `sino_params['angles']` and `sino_params['center_of_rotation']`."""
    angles = sino_params['angles']
    center_of_rotation = sino_params['center_of_rotation']
    num_projections = len(angles)

    # Precompute trigonometric values for all angles
    cos_t = torch.cos(angles).view(1,-1)  # Shape: (1, num_projections)
    sin_t = torch.sin(angles).view(1,-1)  # Shape: (1, num_projections)

    # Compute shifted centers
    x_shifted = centers_x.unsqueeze(1) - center_of_rotation  # Shape: (num_particles, 1)
    y_shifted = centers_y.unsqueeze(1) - center_of_rotation  # Shape: (num_particles, 1)

    # Compute u_particle for all angles and particles
    u_particle = center_of_rotation + x_shifted * cos_t + y_shifted * sin_t  # Shape: (num_particles, num_projections)
    # u_particle = center_of_rotation + centers_r.view(1, -1) * torch.cos(angles.view(-1, 1) - centers_phi.view(1, -1)) #polar version?

    v_particle = centers_z.unsqueeze(1).repeat(1, num_projections)  # Shape: (num_particles, num_projections) - static z
    v_particle *= (0 if sino_params['dimension'] == 2 else 1)
    mag_particle = 0 * u_particle + 1  # no magnification
    # this is where relative orientation of the particles w.r.t. the optical axis can be calculated, but they're symmetric
    return u_particle, v_particle, mag_particle
