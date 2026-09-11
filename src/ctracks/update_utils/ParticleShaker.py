import torch
from lightning import Callback
from src.ctracks.update_utils.ParticleUpdateModule import ParticleUpdateModule


class ParticleShaker(Callback, ParticleUpdateModule):
    """Simulated-annealing-style heuristic: every `interval` epochs, randomly jitters
    particle track and shape parameters by `width` (which decays by `decay` after each
    application). `width` is expected as [track_widths, shape_widths] (each a list of
    per-parameter magnitudes); both are converted to tensors in `on_train_start`."""
    def __init__(self, interval = None, decay = None, width = None):
        Callback.__init__(self)
        ParticleUpdateModule.__init__(self, interval)
        self.decay = decay
        self.width = width

    @torch.no_grad()
    def on_train_start(self, trainer, ct_recon):
        """Convert the configured width lists to tensors on the sample's device."""
        self.width = [torch.tensor(inner_list, dtype=torch.float32, device=ct_recon.ct_sample.device)
                      for inner_list in self.width]

    @torch.no_grad()
    def on_train_epoch_start(self, trainer, ct_recon):
        """Jitter track and shape params in place, then decay the jitter width."""
        if not self.check_interval(trainer.current_epoch):
            return
        print("Shaking particles")
        particles = self.get_particle_component(ct_recon)
        particles.track_model.shake_params(self.width[0])
        particles.shape_model.shake_params(self.width[1])
        self.step()

    def step(self):
        """Decay each width group in place by its corresponding `decay` factor."""
        for i, width_group in enumerate(self.width):
            for j in range(len(width_group)):
                width_group[j] *= self.decay[i]
