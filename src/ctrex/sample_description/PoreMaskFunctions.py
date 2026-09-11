"""Defines pore-space masks that confine particle and volume-model positions to segmented pore
geometry, so components don't get placed or moved into solid grain material.
`PoreMaskFunctions` wraps a real boolean pore-space mask; `PoreMaskPlain` is a permissive stand-in
with no real mask, used when there is no pore geometry to confine against."""

import torch
import scipy.ndimage as snd
import numpy as np
from typing import Tuple
from copy import deepcopy


class PoreMaskFunctions:
    """Confines particle/volume positions to a segmented pore-space mask: a boolean tensor of the
    same shape as the reconstruction volume, where True marks pore (void) space and False marks
    solid grain material. On construction, computes distance-transform maps (`calc_distance_map`)
    that give, for every solid voxel, the nearest pore voxel and distance to it, and vice versa;
    `move_coords_to_pore` and `get_clamped_coordinates` use these maps to snap out-of-pore particle
    coordinates back into the pore space, while `check_particles_in_pore`/`find_edge_detections`
    test particle positions against the mask. `init_pore_mask` further restricts the mask to the
    bounds of any sample components whose shape model requests it (e.g. a cylinder-shaped
    component carving its own geometry out of the mask)."""

    def __init__(self, pore_mask, voi = None, track = None, zero_bounds = (0,0,0),):
        """
        Args:
            pore_mask: boolean tensor of pore (True) / grain (False) space, shape (depth, height, width).
            voi: optional (z, y, x) slice tuple restricting `random_nonzero` sampling to a
                sub-volume; defaults to the full mask extent if not given.
            track: optional track model giving a moving reference frame for `roi_offset`; not
                currently passed by any caller.
            zero_bounds: (z, y, x) margin widths forced to False (grain) at the mask edges, so
                particles can't be placed right at the volume boundary.
        """
        self.pore_mask = pore_mask
        self._voi = voi  # replaces zero bounds
        self.device = pore_mask.device
        self.track = track
        self.__zero_mask_edges(zero_bounds)
        # calculate distance maps

        # self.distance_map, self.nearest_coords, self.boundary_mask, self.distance_map_pore2grain = self.calc_distance_map()
        self.track_model = None

    @property
    def voi(self):
        """Slice tuple (z, y, x) defining the volume-of-interest used by `random_nonzero`;
        lazily defaults to a slice covering the full mask extent if never set explicitly."""
        if self._voi is None:
            self._voi = (slice(0, self.shape[0]),
                         slice(0, self.shape[1]),
                         slice(0, self.shape[2]))
        return self._voi

    @voi.setter
    def voi(self, voi):
        """Also accepts a plain (z, y, x) margin tuple (as with `zero_bounds`) instead of
        slices, in which case it's converted into a centered slice tuple within `shape`."""
        if not hasattr(voi[0], 'start'):
            zero_bounds = voi
            voi = tuple([slice(zero_bounds[di], self.shape[di] - zero_bounds[di]) for di in range(3)])
        self._voi = voi

    @property
    def shape(self):
        return self.pore_mask.shape

    @property
    def voi_shape(self):
        """(depth, height, width) size of the `voi` region."""
        return (self.voi[0].stop - self.voi[0].start,
                self.voi[1].stop - self.voi[1].start,
                self.voi[2].stop - self.voi[2].start)

    @property
    def voi_offset(self):
        """`voi`'s starting offset in (x, y, z) order, matching the coordinate convention used
        elsewhere for particle positions."""
        return [self.voi[2].start, self.voi[1].start, self.voi[0].start]

    def init_pore_mask(self, sample_components):
        """For each sample component whose shape model requests bounds enforcement
        (`shape_model.bounds_particles`), restricts `self.pore_mask` to the component's shape
        bounds at its initial track position (e.g. so a cylinder-shaped component carves its own
        geometry out of the mask). Recomputes the distance maps afterward, and points each
        component's track model back at this mask instance."""
        # e.g. cylinder can adjust pore mask
        for component in sample_components:
            if not component.shape_model.bounds_particles:
                continue
            locations_zyx = torch.nonzero(self.pore_mask)
            in_bounds = torch.tensor(True, device = self.device)
            if (component.track_model.num_particles is not None) and component.shape_model.bounds_particles:
                # position at initial time
                position_start = component.track_model([0])
                in_bounds = component.shape_model.in_bounds(locations_zyx.flip(-1), position_start)
            self.pore_mask[tuple(torch.unbind(locations_zyx, dim = -1))] = in_bounds
        self.calc_distance_map()  # renew nearest pore locations
        for component in sample_components:
            component.track_model.pore_mask = self

    def check_particles_in_pore(self, track_params):
        """Checks whether every control point of `track_params` (N particles, M points, xyz)
        falls inside the pore mask and within the volume bounds; returns a per-particle boolean
        (True only if ALL of its points are valid)."""
        # checks if all positions are in the pore
        depth, height, width = self.shape

        x_coords = track_params[:, :, 0].long()
        y_coords = track_params[:, :, 1].long()
        z_coords = track_params[:, :, 2].long()

        # check in bounds
        valid_z = (z_coords >= 0) & (z_coords < depth - 1)
        valid_y = (y_coords >= 0) & (y_coords < height - 1)
        valid_x = (x_coords >= 0) & (x_coords < width - 1)
        valid_coords = valid_z & valid_y & valid_x

        clipped_z = torch.clamp(z_coords, 0, depth - 1)
        clipped_y = torch.clamp(y_coords, 0, height - 1)
        clipped_x = torch.clamp(x_coords, 0, width - 1)

        mask_values = self.pore_mask[clipped_z, clipped_y, clipped_x]

        combined_check = mask_values & valid_coords

        # Check if all M positions are True for each particle
        all_true = torch.all(combined_check, dim=1)
        return all_true

    @property
    def roi_offset(self):
        """Offset (xyz) between `self.track`'s position at time 0 and the mask's geometric
        center, used to re-center coordinates onto a moving reference frame; 0 if no `track`
        was supplied (the case for every current caller)."""
        if self.track is not None:
            # position at start time
            centre_coordinates = self.track(torch.tensor([0], device = self.device))
            shape_centre = torch.as_tensor(self.shape[::-1], device=self.device) / 2
            roi_offset = centre_coordinates - shape_centre
        else:
            roi_offset = 0
        return roi_offset

    def get_clamped_coordinates(self, coords_xyz, replace = False):
        """Floors `coords_xyz` (after subtracting `mask_roi_offset`) to integer voxel indices
        and clamps them into the mask bounds. If `replace` is True, overwrites `coords_xyz` in
        place with the clamped integer position plus its original fractional part (preserving
        gradients w.r.t. the sub-voxel offset).

        Returns:
            A (z, y, x) tuple of long tensors, matching the mask's own indexing order.
        """
        coords_xyz = coords_xyz - mask_roi_offset(self)
        max_bounds = torch.tensor(self.shape[::-1], dtype = coords_xyz.dtype, device = coords_xyz.device) - 1
        for i in range(coords_xyz.ndim - 1):
            max_bounds = max_bounds.unsqueeze(0)
        coords_xyz_sample = torch.clamp(torch.floor(coords_xyz[...,:3]), 0 * coords_xyz, max_bounds).long()
        if replace:
            decimals = coords_xyz - torch.floor(coords_xyz)
            coords_xyz[:] = coords_xyz_sample + decimals
        return torch.unbind(coords_xyz_sample, dim = -1)[::-1]  # zyx order

    def move_coords_to_pore(self, coords_xyz):
        """Snaps `coords_xyz` (xyz) that fall outside the pore mask back onto the nearest pore
        voxel, using the precomputed `boundary_mask` correction vectors, then re-centers by
        `roi_offset`. Modifies and returns `coords_xyz` in place."""
        coords_zyx_sample = self.get_clamped_coordinates(coords_xyz, replace = True)
        corrections = self.boundary_mask[coords_zyx_sample]  # zyx order
        corrections = torch.flip(corrections, [-1]).float()
        coords_xyz[:] = coords_xyz + corrections + self.roi_offset
        return coords_xyz

    def find_edge_detections(self, track_params, edge_range=1):
        """For each particle track in `track_params` (N particles, M points, xyz), flags whether
        any of its points lies within `edge_range` voxels of the pore/grain boundary (via
        `distance_map_pore2grain`). Used to identify particles that should be dropped or
        re-initialized because they've drifted onto the pore boundary."""
        # particle coords N,M,3
        # edge range - how many voxels away from boundary is edge

        N, M, _ = track_params.shape

        # find tracks that have points in mask - automatic removal
        idx = torch.floor(track_params).long()

        flat_z = idx[..., 2].reshape(-1)
        flat_y = idx[..., 1].reshape(-1)
        flat_x = idx[..., 0].reshape(-1)

        dist_to_boundary_flat = self.distance_map_pore2grain[flat_z, flat_y, flat_x]
        dist_to_boundary = dist_to_boundary_flat.view(N, M)

        edge_coords = dist_to_boundary < edge_range
        edge_particles = edge_coords.any(dim=1)

        return edge_particles

    def random_nonzero(self, num_particles):
        """Draws `num_particles` random pore-space (mask value True) voxel centers (xyz) from
        within `self.voi`, without replacement."""
        centers_xyz = torch.flip(torch.nonzero(self.pore_mask[self.voi]), [-1])
        centers_xyz += torch.tensor([self.voi[2].start, self.voi[1].start, self.voi[0].start],
                                    dtype = centers_xyz.dtype, device=centers_xyz.device)

        # centers_xyz = torch.tensor(torch.flip(torch.where(init_mask, ),[0]))
        num_voxels = len(centers_xyz)
        random_indices = torch.randperm(num_voxels, device=centers_xyz.device)[:num_particles]
        centers_xyz = centers_xyz[random_indices].to(torch.float32)
        return centers_xyz

    def __zero_mask_edges(self, widths):
        """Forces a margin of width `widths[dim]` (z, y, x) at both ends of every axis to
        False (grain), so particles/mask features can't sit right at the mask boundary."""
        shape = self.shape
        ndim = len(shape)
        for dim in range(ndim):
            width = widths[dim]
            if width > shape[dim]: continue #skip if zero width > dimension of space
            lower_slice = [slice(None)] * ndim
            upper_slice = [slice(None)] * ndim
            lower_slice[dim] = slice(0, width)
            upper_slice[dim] = slice(shape[dim] - width, shape[dim])
            self.pore_mask[tuple(lower_slice)] = False
            self.pore_mask[tuple(upper_slice)] = False
        return self.pore_mask

    def calc_distance_map(self):
        """Computes and caches the Euclidean distance transforms needed to snap particles into
        the pore space: `distance_map`/`nearest_coords` (grain voxel -> distance/coords of the
        nearest pore voxel), `distance_map_pore2grain` (the reverse: pore voxel -> distance to
        nearest grain), and `boundary_mask` (per-voxel correction vector to the nearest pore
        voxel, used by `move_coords_to_pore`). Falls back to an all-zero distance map (and
        infinite `distance_map_pore2grain`) if the mask has no grain (solid) voxels at all."""
        inverted_pore_mask = (~self.pore_mask).cpu().numpy()
        voxel_coordinates = np.array(np.meshgrid(*[np.arange(0, dsize) for dsize in self.pore_mask.shape],
                                                 indexing='ij'))
        if np.any(inverted_pore_mask):
            self.distance_map, self.nearest_coords = snd.distance_transform_edt(inverted_pore_mask, return_distances=True,
                                                                                return_indices=True)
            pore_mask_np = (self.pore_mask).cpu().numpy()
            self.distance_map_pore2grain = snd.distance_transform_edt(pore_mask_np, return_distances=True)
            self.distance_map_pore2grain = torch.from_numpy(self.distance_map_pore2grain).to(self.device)
        else:
            self.distance_map = inverted_pore_mask * 0.
            self.nearest_coords = voxel_coordinates  # (3: zyx, nz, ny, nx)
            self.distance_map_pore2grain = self.pore_mask * torch.inf

        # calculating boundary mask
        self.boundary_mask = self.nearest_coords.copy()
        self.boundary_mask -= voxel_coordinates
        self.boundary_mask = np.rollaxis(self.boundary_mask, 0, self.pore_mask.ndim + 1)

        self.distance_map = torch.from_numpy(self.distance_map).to(self.device)
        self.nearest_coords = torch.from_numpy(self.nearest_coords).to(self.device)
        self.boundary_mask = torch.from_numpy(self.boundary_mask).to(self.device)
        self.max_distance = self.distance_map.max()

        return self.distance_map, self.nearest_coords, self.boundary_mask, self.distance_map_pore2grain

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Allows this object to be used directly as a tensor argument in `torch.*` operations,
        by substituting `self.pore_mask` for any `PoreMaskFunctions` instance found among the
        call's arguments before dispatching."""
        if kwargs is None:
            kwargs = {}

        # Replace any occurrences of an *instance* of cls with the instance's pore_mask
        new_args = []
        for arg in args:
            if isinstance(arg, cls):
                new_args.append(arg.pore_mask)  # Access pore_mask from the instance
            else:
                new_args.append(arg)

        try:
            return func(*new_args, **kwargs)
        except Exception as e:
            raise NotImplementedError(f"Function '{func}' not supported by pore mask.") from e

    __torch_function__ = __torch_dispatch__

    def __getitem__(self, indices):
        """Indexes directly into the underlying `pore_mask` tensor."""
        return self.pore_mask[indices]

    def __array__(self):
        """Supports `np.asarray()`/NumPy interop by returning `pore_mask` as a CPU NumPy array."""
        return self.pore_mask.cpu().numpy()


class PoreMaskPlain(PoreMaskFunctions):
    """Permissive stand-in for `PoreMaskFunctions` with no real pore-space mask: used when a
    sample component needs a `pore_mask`-shaped object (for its `shape`/`voi` bounds) but there
    is no actual pore geometry to confine against, e.g. plain volume/matrix reconstructions (see
    `recon_simulated_volume.py`, `templates.py`). Positions are only bounded by the volume shape
    (or an optional `voi` sub-region of it); `move_coords_to_pore`, `find_edge_detections`, and
    `check_particles_in_pore` are all no-ops that never move or reject a particle."""

    # noinspection PyMissingConstructor
    def __init__(self, shape, device, voi = None):
        self.shape = shape
        self.device = device
        self.pore_mask = None
        self.track = None
        self._voi = voi

    def __copy__(self):
        """ https://stackoverflow.com/questions/1500718/how-to-override-the-copy-deepcopy-operations-for-a-python-object"""
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """ https://stackoverflow.com/questions/1500718/how-to-override-the-copy-deepcopy-operations-for-a-python-object"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    def init_pore_mask(self, sample_components):
        """No-op: there's no real mask to restrict here."""
        pass

    def calc_distance_map(self):
        """No-op: no distance maps to compute without a real mask."""
        pass

    def get_clamped_coordinates(self, coords_xyz, replace = False):
        """Same as `PoreMaskFunctions.get_clamped_coordinates`, clamping into `self.shape`
        bounds only (there's no real mask to snap onto)."""

        coords_xyz = coords_xyz - mask_roi_offset(self)
        max_bounds = torch.tensor(self.shape[::-1], dtype = coords_xyz.dtype, device = coords_xyz.device) - 1
        for i in range(coords_xyz.ndim - 1):
            max_bounds = max_bounds.unsqueeze(0)
        coords_xyz_sample = torch.clamp(torch.floor(coords_xyz[...,:3]), 0 * coords_xyz, max_bounds).long()
        if replace:
            decimals = coords_xyz - torch.floor(coords_xyz)
            coords_xyz[:] = coords_xyz_sample + decimals
        return torch.unbind(coords_xyz_sample, dim = -1)[::-1]  # zyx order

    def move_coords_to_pore(self, coords_xyz):
        """No-op: returns `coords_xyz` unchanged (there's no pore mask to snap onto)."""
        # Todo: this breaks the particles' grad
        # coords_zyx_sample = self.get_clamped_coordinates(coords_xyz, replace = True)
        # coords_zyx_sample = torch.stack(coords_zyx_sample[::-1], dim = -1).float()
        # coords_xyz[:] = coords_zyx_sample + self.roi_offset
        return coords_xyz

    def find_edge_detections(self, track_params, edge_range=1):
        """No-op: always reports that no particles are near an edge (there's no mask boundary)."""
        return torch.zeros(len(track_params), dtype = torch.bool, device = self.device)


    def check_particles_in_pore(self, track_params):
        """No-op: always reports every particle as valid (there's no mask to check against)."""
        return torch.ones(len(track_params), dtype = torch.bool, device = self.device)

    def random_nonzero(self, num_particles):
        """Draws `num_particles` uniformly random coordinates (xyz) within `self.voi` (or the
        full volume), rather than sampling actual pore voxels - since there's no mask to sample
        from."""
        centers_xyz = torch.rand((num_particles, 3), device=self.device)
        centers_xyz *= (torch.tensor(self.voi_shape[::-1], device=self.device).unsqueeze(0))
        centers_xyz += torch.tensor(self.voi_offset, device = self.device).unsqueeze(0)
        return centers_xyz

def mask_roi_offset(mask):
    """Offset (xyz) to re-center coordinates onto `mask.track_model`'s position at time 0, if
    one has been attached to `mask` (expected as a (track_model, track_params) pair); 0
    otherwise - which is the case for every current caller, since no pore mask instance
    currently has `track_model` set to anything but `None`."""
    if hasattr(mask, 'track_model') and mask.track_model is not None:
        track_model, track_params = mask.track_model
        # position at start time
        centre_coordinates = track_model(track_params, torch.tensor([0], device = mask.device))
        shape_centre = torch.as_tensor(mask.shape[::-1], device=mask.device) / 2
        roi_offset = centre_coordinates - shape_centre
    else:
        roi_offset = 0
    return roi_offset
