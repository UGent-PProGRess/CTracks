import torch
from lightning import Callback

from ctrex.optimization.reconstruction import CTReconstruction


class Scheduler(Callback):
    """Wraps a torch LR scheduler (e.g. ReduceLROnPlateau) as a Lightning callback: the
    scheduler is constructed in `on_train_start` (once the optimizer exists) and stepped
    each epoch in `on_train_epoch_end` using the logged data loss."""
    def __init__(self, schedule_function = None, mode = 'min', factor = 0.5, patience = 20):
        super().__init__()
        self.schedule_function = schedule_function
        self.scheduler = None
        self.mode = mode
        self.factor = factor
        self.patience = patience

    @torch.no_grad()
    def on_train_start(self, trainer, ct_recon: CTReconstruction):
        """Instantiate the wrapped scheduler around the reconstruction's optimizer."""
        if self.schedule_function is not None:
            optimizer = ct_recon.optimizers()
            self.scheduler = self.schedule_function(optimizer, mode = self.mode, factor = self.factor,
                                                    patience = self.patience)

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, ct_recon):
        """Step the scheduler using the epoch's logged data loss metric - unless training was
        frozen this epoch (e.g. by ParticleAddition, to measure unexplained signal without the
        model changing under it), in which case there's no real loss to compare against, so we
        step the scheduler with an explicit 0 instead. With mode='min', ReduceLROnPlateau treats
        0 as an unbeatable improvement: it won't anneal on the frozen epoch itself, but pins its
        "best" at 0, so every subsequent real (necessarily positive) loss counts as "no
        improvement" - after `patience` such epochs the LR anneals, and since best can never be
        beaten again, it keeps annealing every `patience` epochs from then on. This is a
        deliberate, explicit choice to get that annealing cadence, made here instead of via the
        previous behaviour, which produced the exact same scheduler-side effect only as an
        unintended side effect of also logging that fake 0 to the real epoch/data_loss metric
        (corrupting the visible loss curve, e.g. in TensorBoard) - stepping the scheduler
        directly, without also logging, keeps the behaviour but not the corrupted logging."""
        if getattr(ct_recon, 'epoch_was_frozen', False):
            self.scheduler.step(0)
        else:
            data_loss = trainer.callback_metrics["epoch/data_loss"]
            self.scheduler.step(data_loss)
