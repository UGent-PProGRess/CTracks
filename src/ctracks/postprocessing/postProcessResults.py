import torch
from lightning import Callback


class ParticlePostProcess(Callback):
    """Lightning callback that prunes reconstructed particles once, at the end of training.

    Removes particles near the pore/grain boundary, optionally removes dark/thin particles
    (via `remove_dark_particles`/`remove_thin_particles` below), and optionally removes
    particles whose velocity deviates too far from their local spatial neighborhood's mean
    velocity.
    """

    def __init__(self, edge_range = 0, att_removal_fraction = 0, thin_removal_fraction = 0, thin_rad_thresh = None,
                 vel_cluster_range = 0, vel_difference_threshold = 1, vel_min_neighbours = 3):
        super().__init__()
        self.edge_range = edge_range
        self.att_removal_fraction = att_removal_fraction
        self.thin_removal_fraction = thin_removal_fraction
        self.thin_rad_thresh = thin_rad_thresh
        self.vel_cluster_range = vel_cluster_range
        self.vel_difference_threshold = vel_difference_threshold
        self.vel_min_neighbours = vel_min_neighbours

    def on_train_end(self, trainer, ct_recon):
        """PyTorch Lightning hook, invoked by the Trainer once training finishes. Runs the
        configured particle-pruning steps in sequence on the reconstructed sample."""
        print("Post processing particles")
        particles = ct_recon.ct_sample.projectors.particles.sample_component
        pore_mask = ct_recon.ct_sample.projectors.particles.sample_component.track_model.pore_mask

        edge_particles = pore_mask.find_edge_detections(particles.track_model.control_points, self.edge_range)
        particles.remove_particles(edge_particles)
        print(f"Removed {edge_particles.sum().item()} edge detections.")

        if self.att_removal_fraction > 0:
            remove_dark_particles(particles, fraction = self.att_removal_fraction)

        if self.thin_removal_fraction > 0:
            remove_thin_particles(particles, rad_thresh = self.thin_rad_thresh, fraction = self.thin_removal_fraction)

        if self.vel_cluster_range > 0 and particles.track_model.num_control_points > 0:
            filter_velocity_clusters(particles, self.vel_cluster_range, self.vel_difference_threshold,
                                     self.vel_min_neighbours)


def remove_thin_particles(particle_component, optimizer=None, rad_thresh=None, fraction=1):
    """Remove a `fraction` of the particles thinner than `rad_thresh * rad_min` (or all
    particles, if `rad_thresh` is None), preferring to keep the brightest (highest
    attenuation) ones among those below threshold. Mutates `particle_component` in
    place. Returns whether any particles were removed."""
    particle_shape_model = particle_component.shape_model  # type: SphereShape
    rad, att = particle_shape_model.shape_params
    rad_min = particle_shape_model.rad_min
    if rad_thresh is None: thin_mask = torch.ones(len(rad), device = rad.device, dtype = torch.bool)
    else: thin_mask = rad < rad_thresh*rad_min  #remove particles below threshold, scaled by min radius

    #only removing a fraction of thin particles
    true_indices = thin_mask.nonzero(as_tuple=True)[0]
    num_thin = len(true_indices)
    num_to_keep = int(num_thin * (1-fraction))
    if num_thin - num_to_keep > 0:
        sorted_indices = torch.argsort(att[true_indices], descending = True)
        indices_to_keep = true_indices[sorted_indices[:num_to_keep]]
        thin_mask[indices_to_keep] = False
        particle_component.remove_particles(thin_mask, optimizer)
        print(f"Removed {num_thin - num_to_keep} thin particles")
        return True

    return False


def remove_dark_particles(particle_component, optimizer=None, fraction=0.1):
    """Remove the darkest `fraction` of particles (lowest attenuation first), breaking
    ties by removing the smaller-radius particle first. Mutates `particle_component`
    in place."""
    rad, att = particle_component.shape_model.shape_params
    num = particle_component.shape_model.num_particles
    num_to_remove = int(torch.ceil(torch.tensor(num*fraction, device = att.device)).item())
    if num_to_remove == 0:
        mask = torch.zeros(num, dtype=torch.bool, device = att.device)
    elif num_to_remove >= num:
        mask = torch.ones(num, dtype=torch.bool, device = att.device)
    else:
        rad_order = torch.argsort(rad, descending = False)
        att_reorder = att[rad_order]
        att_sort_rad_order = torch.argsort(att_reorder, descending = False, stable = True )
        sorted_indices = rad_order[att_sort_rad_order]
        darkest_indices = sorted_indices[:num_to_remove]

        mask = torch.zeros_like(att, dtype=torch.bool)
        mask[darkest_indices] = True

    print(f"Removing {torch.sum(mask)} dark particles")
    particle_component.remove_particles(mask, optimizer)


def filter_velocity_clusters(particles, neighbour_distance=15, threshold=1, min_neighbours = 3):
    """Remove particles whose velocity deviates from the mean velocity of their local spatial
    neighborhood by more than ``threshold``, considering only neighborhoods of at least
    ``min_neighbours`` particles within ``neighbour_distance``."""
    tracks = particles.track_model.control_points
    velocities = (tracks[:, 1:, :] - tracks[:, :-1, :]).mean(dim=1)

    mid_points = tracks.mean(dim=1)
    distances = torch.cdist(mid_points, mid_points)
    neighbors_mask = distances < neighbour_distance  # (N, N)

    masked_velocities = velocities.unsqueeze(0) * neighbors_mask.unsqueeze(-1)  # (N, N, 3)
    num_neighbors = neighbors_mask.sum(dim=1).float()

    divisor = num_neighbors.clone().unsqueeze(-1)
    divisor[divisor == 0] = 1.0
    local_mean_velocity = masked_velocities.sum(dim=1) / divisor  # (N, 3)

    velocity_diff = velocities - local_mean_velocity  # (N, 3)
    error_magnitude = velocity_diff.norm(dim=1)  # (N)

    is_high_error = error_magnitude > threshold
    is_large_cluster = num_neighbors >= min_neighbours
    remove_mask = is_high_error & is_large_cluster
    print(f"Removing {remove_mask.sum().item()} particles from velocity clustering.")

    # NOTE: Assuming ct_sample.remove_particles accepts a boolean mask
    particles.remove_particles(remove_mask)
