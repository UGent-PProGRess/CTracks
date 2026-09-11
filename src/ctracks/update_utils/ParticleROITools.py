import torch
from lightning import Callback

from src.ctracks.update_utils.ParticleUpdateModule import ParticleUpdateModule
import ctrex.sample_description.track_models as tm

class ParticleROITools(Callback, ParticleUpdateModule):
    """Periodically removes particles whose track never projects into the detector ROI
    over the full scan, and respawns an equal number of new particles nearby. A no-op
    unless the detector actually has a restricted ROI (checked in `on_train_start`)."""
    def __init__(self, interval = 1):
        Callback.__init__(self)
        ParticleUpdateModule.__init__(self, interval)
        self.running = False
        self.roi_bounds = []

    @torch.no_grad()
    def on_train_start(self, trainer, ct_recon):
        """Detect whether the detector ROI is actually restricted (vs. the full detector)
        and, if so, enable updates and record ROI-corrected (u, v) bounds for later use."""
        detector = ct_recon.ct_sample.trajectory.detector
        size = detector.height, detector.width
        roi = detector.roi
        for i, s in enumerate(roi): #Check if ROI is actually set
            if s.start is not None or s.stop is not None:
                if s.start > 0 or s.stop < size[i]:
                    self.running = True
                    for s in roi: #projected coordinates are roi corrected so need to correct here too
                        start = 0
                        stop = s.stop -  s.start
                        self.roi_bounds.append((start, stop))
                    break


    @torch.no_grad()
    def on_train_epoch_end(self, trainer, ct_recon):
        """Project every particle track across all detector views; remove any particle that
        never lands inside the ROI, and add back an equal number of new particles nearby."""
        if not self.check_interval(trainer.current_epoch): return
        particles = self.get_particle_component(ct_recon)
        trajectory = ct_recon.ct_sample.trajectory

        u_min, u_max = self.roi_bounds[1]
        v_min, v_max = self.roi_bounds[0]


        sampled_projections = torch.arange(0, len(trajectory.angles), device=trajectory.device, dtype=torch.long)

        sampled_projections = trajectory.projection_indices(sampled_projections)
        views = trajectory.calc_trajectory(sampled_projections)
        projection_times = trajectory.projection_time(sampled_projections)
        centers_time = particles.track_model(projection_times)
        u, v, _ = trajectory.project_voxels(centers_time, views)  # N, numprojections

        u_in = torch.ge(u, u_min) & torch.lt(u, u_max)
        v_in = torch.ge(v, v_min) & torch.lt(v, v_max)

        is_inside_point = u_in & v_in

        keep_particle = torch.any(is_inside_point, dim=1)
        remove_mask = torch.logical_not(keep_particle)
        num_particles_removed = remove_mask.sum().item()
        optimizer = ct_recon.optimizers()
        particles.remove_particles(remove_mask, optimizer)
        particles.add_particles(num_particles_removed, optimizer, init_mode_tracks=tm.init_random_nearby_vel)
