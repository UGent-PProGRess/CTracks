from lightning import Callback
import torch
from ctracks.particle_projectors import ParticleCTSimulation
from ctrex.sample_description import track_models as tm
from src.ctracks.update_utils.ParticleUpdateModule import ParticleUpdateModule


class ParticleAddition(Callback, ParticleUpdateModule):
    """Adaptive particle-count controller. Every `interval` epochs, freezes gradient
    updates for one epoch, measures per-subset unexplained signal (measured minus
    simulated sinogram, relative to the mean per-particle intensity), and adds new
    particles if the unexplained signal exceeds `intensity_threshold`."""

    def __init__(self, interval = None, intensity_threshold = None):
        Callback.__init__(self)
        ParticleUpdateModule.__init__(self, interval)
        self.intensity_threshold = intensity_threshold
        self.update_average_intensity = True
        self.sample_intensities = None  # num subsets,2 - sinogram intensity differences, mean particle intensity


    @torch.no_grad()
    def on_train_start(self, trainer, ct_recon):
        """Allocate the per-subset (intensity difference, mean particle intensity) buffer."""
        num_subsets = ct_recon.sampler.num_subsets
        self.sample_intensities = torch.zeros((num_subsets, 2), device=ct_recon.ct_sample.device,
                                              dtype=torch.float32)

    @torch.no_grad()
    def on_train_epoch_start(self, trainer, ct_recon):
        """On interval epochs, freeze gradient updates for the epoch so the measured
        signal reflects the current fit rather than a changing one."""
        self.update_batches = self.check_interval(trainer.current_epoch)
        if self.update_batches:
            trainer.should_stop = False
            ct_recon.training_freeze = True
            self.sample_intensities[:,0] = 0


    @torch.no_grad()
    def on_train_batch_end(self, trainer, ct_recon, outputs, batch, subiteration):
        """Accumulate per-subset intensity statistics while updates are frozen."""
        if self.update_batches:
            sampled_projections = outputs['sampled_projections']
            simulated_sinogram_sample = outputs['simulated_sinogram_sample']
            sinogram_sample = ct_recon.sinogram[:, sampled_projections]

            si = subiteration % ct_recon.sampler.num_subsets
            self.calc_average_particle_intensity(ct_recon, sinogram_sample, sampled_projections, si)
            self.calc_sample_intensity_difference(sinogram_sample, simulated_sinogram_sample, si)


    @torch.no_grad()
    def on_train_epoch_end(self, trainer, ct_recon):
        """Act on the accumulated statistics, then unfreeze gradient updates."""
        if self.update_batches:
            self.update_particles(ct_recon)

        # reset for optimization
        self.update_batches = False
        ct_recon.training_freeze = False

    @torch.no_grad()
    def calc_average_particle_intensity(self, ct_recon, sinogram_sample, sampled_projections,
                                    sample_iteration):
        """Compute the mean per-particle sinogram intensity for this subset (cached in
        `self.sample_intensities` after all subsets have been seen once, since it's
        expensive to recompute and doesn't change quickly). If a noise projector is
        present, instead computes the significant (median-absolute-deviation thresholded)
        unexplained signal relative to that mean intensity."""
        if not self.update_average_intensity: return

        particle_sinogram = 0 * sinogram_sample
        num_particles = 0

        trajectory = ct_recon.ct_sample.trajectory
        sampled_projections = trajectory.projection_indices(sampled_projections)
        views = trajectory.calc_trajectory(sampled_projections)

        projectors = ct_recon.ct_sample.projectors

        for name, module in projectors.projector_children():
            if type(module) is ParticleCTSimulation:
                particle_sinogram = module.forward(particle_sinogram, views, sampled_projections)
                num_particles += module.sample_component.num_particles

        particle_intensity = torch.sum(particle_sinogram) / num_particles

        if hasattr(projectors, "noise"):
            particle_sinogram = projectors.noise.forward(particle_sinogram, views, sampled_projections)
            difference_image = sinogram_sample - particle_sinogram
            D_flat = difference_image.flatten()
            D_median = torch.median(D_flat)
            abs_deviations = torch.abs(D_flat - D_median)
            mad = torch.median(abs_deviations)
            sigma_D = mad * 1.4826
            T = 4 * sigma_D  # TODO tune this value
            difference_mask = torch.abs(difference_image) > T
            signal_diff = torch.sum(torch.abs(difference_image) * difference_mask.float())
            particle_intensity = signal_diff / particle_intensity


        self.sample_intensities[sample_iteration][1] = particle_intensity

        if sample_iteration == len(self.sample_intensities[:, 0]) - 1: self.update_average_intensity = False

    def calc_sample_intensity_difference(self, sinogram_sample, simulated_sinogram_sample, sample_iteration):
        """Store the unexplained signal for this subset as a fraction of its mean
        per-particle intensity, in `self.sample_intensities[sample_iteration][0]`."""
        intensity_difference = torch.sum(sinogram_sample - simulated_sinogram_sample)
        difference_fraction = intensity_difference / self.sample_intensities[sample_iteration][1]
        self.sample_intensities[sample_iteration][0] = difference_fraction

    def update_particles(self, ct_recon):
        """If the mean unexplained-signal fraction across subsets exceeds
        `intensity_threshold`, add new particles when that signal is positive (undersimulated
        regions) — the number added scales with how far over the threshold it is.
        When negative (oversimulated regions), particle removal would be the correct
        response but is currently disabled; nothing happens in that case. Returns whether
        the threshold was exceeded (not whether particles were actually added)."""
        particles = self.get_particle_component(ct_recon)
        mean_intensity_diff = torch.mean(self.sample_intensities[:,0])
        difference_fraction_threshold = torch.abs(mean_intensity_diff) / self.intensity_threshold
        if difference_fraction_threshold < 1: return False
        n_particles = max(1, int(difference_fraction_threshold.floor()))
        if mean_intensity_diff > 0:
            print(f"Adding {n_particles} particles")
            optimizer = ct_recon.optimizers()
            self._add_particles(n_particles, particles, optimizer)
        else:
            # Removal of particles has been temporarily disabled - looking for new options
            print(f"Removing particles disabled - trying to remove {n_particles} particles")

        return True

    def _add_particles(self, n_particles, particles, optimizer):
        """Add n_particles new particles, initialized with nearby velocities."""
        particles.add_particles(n_particles, optimizer, init_mode_tracks = tm.init_random_nearby_vel) #new particles take nearby velocities


class ParticleAddition_fixed(Callback, ParticleUpdateModule):
    """Simpler alternative to `ParticleAddition`: adds a fixed number of particles every
    `interval` epochs, unconditionally (no unexplained-signal measurement), with the
    amount decaying by `decay` after each addition."""
    def __init__(self, interval=None, amount = None, decay=None):
        Callback.__init__(self)
        ParticleUpdateModule.__init__(self, interval)
        self.decay = decay
        self.amount = amount


    @torch.no_grad()
    def on_train_epoch_start(self, trainer, ct_recon):
        """On interval epochs, unconditionally add `self.amount` new particles, then decay
        `self.amount` for next time."""
        if self.check_interval(trainer.current_epoch):
            print(f"Adding {self.amount} particles")
            particles = self.get_particle_component(ct_recon)
            particles.add_particles(self.amount, ct_recon.optimizers(),
                                    init_mode_tracks=tm.init_random_nearby_vel)  # new particles take nearby velocities
            self.step()

    def step(self):
        """Decay `self.amount` in place by `self.decay`."""
        self.amount = int(self.amount*self.decay)

