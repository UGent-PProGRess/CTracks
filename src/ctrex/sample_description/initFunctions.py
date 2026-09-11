"""Particle position/track initialization helpers: random and pore-mask-aware sampling of new
particle centers, and "close" (near-ground-truth) initialization used for synthetic-data
reconstruction, all operating in xyz sample-space coordinates."""

import torch
from ctrex.sample_description.PoreMaskFunctions import PoreMaskFunctions
from scipy.spatial import KDTree


def random_coordinates_mask(num_particles, pore_mask: PoreMaskFunctions, device):
    """Draws `num_particles` random particle centers (xyz) from pore-space voxels of `pore_mask`
    (see `PoreMaskFunctions.random_nonzero`), each nudged by a random sub-voxel offset (only along
    axes where the mask has more than one voxel), and re-centered by `pore_mask.roi_offset`."""
    roi_offset = pore_mask.roi_offset
    centers_xyz = pore_mask.random_nonzero(num_particles)
    # add some random decimal value between 0 and 1 if xyz dimension size is not 1
    decimals = torch.rand((num_particles, 3), device=device)
    decimals *= (1 != torch.tensor(pore_mask.shape[::-1], device=device).unsqueeze(0))
    centers_xyz += decimals
    centers_xyz += roi_offset
    return centers_xyz  # (num_particles, 3 -> xyz)


def random_coordinates_volume(num_particles, vol_shape, device, dims = 3):
    """Draws `num_particles` random particle centers (xyz) uniformly within `vol_shape`. If
    `dims == 2`, fixes the z-coordinate to the mid-plane instead of randomizing it."""
    centers_xyz = torch.rand(num_particles,3) * torch.as_tensor(vol_shape[::-1])
    if dims == 2:
        centers_xyz[:,2] = vol_shape[0] / 2
    return centers_xyz.to(device)


def close_coordinates_mask(coords_xyz, sigs, pore_mask):
    """Perturbs `coords_xyz` by a uniform random offset in [-sigs, sigs] per axis (skipped if
    `sigs` is None), then snaps the result into the pore mask via `pore_mask.move_coords_to_pore`
    (skipped if `pore_mask` is None)."""
    num_particles = len(coords_xyz)
    # uniform between (-1 and 1) times sigs
    if sigs is not None:
        coords_xyz = coords_xyz + (torch.rand(num_particles, 3, device=pore_mask.device) * 2 - 1) * sigs.unsqueeze(0)
    if pore_mask is not None:
        coords_xyz = pore_mask.move_coords_to_pore(coords_xyz[...,:3])  # TODO update this to move along displacement
    return coords_xyz  # (num_particles,3)


def close_control_points(track_params, pore_mask, sigs=None):
    """Builds a full set of track control points close to the ground-truth `track_params` (N
    tracks, M control points, xyz(+extra)), for initializing a reconstruction near a
    known/simulated answer. The first and last control points are placed near their
    ground-truth counterparts (via `close_coordinates_mask`, perturbed by `sigs`); any
    intermediate control points are placed by linearly interpolating from the first point
    towards the (already-perturbed) last point, with decreasing perturbation as points
    approach the end. In the 2D case (`pore_mask.shape[0] == 1`), the z-coordinate is zeroed
    out. Preserves `track_params.requires_grad` on the output; does not enable it.

    Args:
        track_params: ground-truth control points, shape (N, M, >=3).
        pore_mask: pore mask used to keep perturbed points off grain material.
        sigs: per-axis perturbation magnitude for the first control point; the perturbation
            shrinks for later points as they approach the last one.
    """
    # Initialise centers close to the ground truth position
    # sigs of same shape
    num_tracks, num_control_points, ndims = track_params.shape
    if sigs is None:
        sigs = torch.zeros(ndims, dtype=torch.float32, device=track_params.device)
    else:
        sigs = torch.as_tensor(sigs, dtype=torch.float32, device=track_params.device)
        # ensure length of sigs is sufficient
    recon_tracks = torch.ones_like(track_params, requires_grad=False)
    with torch.no_grad():
        recon_tracks[:, 0, :3] = close_coordinates_mask(track_params[:, 0, :3], sigs, pore_mask)
        if num_control_points >= 2:
            # keep total displacement roughly equal
            displacement = track_params[:, -1] - track_params[:, 0]
            recon_tracks[:, -1, :3] = close_coordinates_mask(recon_tracks[:, 0, :3] + displacement, sigs, pore_mask)
            # progress towards end point, using close_coordinates_mask to avoid grains
            for ci in range(1, num_control_points - 1):  # control_index
                segment = (recon_tracks[:, -1, :3] - recon_tracks[:, ci - 1, :3]) / (num_control_points - ci)
                recon_tracks[:, ci] = close_coordinates_mask(recon_tracks[:, 0, :3] + segment,
                                                             sigs / (num_control_points - ci), pore_mask)
    # recon_tracks.to(track_params.device)
        if pore_mask.shape[0] == 1:  # 2D case
            # recon_tracks[:,:,2].requires_grad = False
            recon_tracks[:, :, 2] *= 0
    recon_tracks.requires_grad = track_params.requires_grad #only enable grad if input tracks also have grad - set grad on elsewhere if initialising


    return recon_tracks

def nearby_velocity_init(centers_xyz, existing_params, pore_mask, num_neighbours = 5):
    """Initializes a second control point for each of `centers_xyz` by carrying over the mean
    recent velocity of its `num_neighbours` nearest existing particles (by mean position, via
    `find_nearest_neighbors`), then snapping the result into the pore mask.

    Args:
        centers_xyz: starting positions (xyz) of the new particles, shape (N, 3).
        existing_params: control points of already-initialized particles, shape (K, M, 3),
            used as a velocity reference.
        pore_mask: pore mask used to keep the projected position off grain material.
        num_neighbours: number of nearest existing particles to average the velocity over.
    """
    neighbor_indices = find_nearest_neighbors(centers_xyz, existing_params, k=num_neighbours) #N,k
    neighbor_particles = existing_params[neighbor_indices] #N,k,M,3

    #linear displacement
    displacements = neighbor_particles[:,:,-1,:] - neighbor_particles[:,:,0,:] #N,k,3
    mean_displacement = torch.mean(displacements, dim = 1) #N,3

    centers_xyz2 = centers_xyz + mean_displacement
    centers_xyz2 = pore_mask.move_coords_to_pore(centers_xyz2[..., :3])

    return centers_xyz2

def find_nearest_neighbors(centers_xyz, existing_params, k=1):
    """Finds, for each of `centers_xyz`, the indices of its `k` nearest existing particles in
    `existing_params` by mean position (KD-tree query on CPU)."""
    mean_positions = torch.mean(existing_params, dim = 1).to('cpu').numpy() #N,3
    tree = KDTree(mean_positions)
    distances, indices = tree.query(centers_xyz.to('cpu').numpy(), k=min(k, existing_params.shape[0]))  # check distances against a threshold? just use flow direction if too far?
    return indices
