"""Orchestrates ordered-subset (OS) CT reconstruction as a PyTorch Lightning `LightningModule`.

Defines `CTReconstruction`, which wraps a `CTSimulation` sample model and a `SinogramSampler`
so that Lightning's `Trainer` drives OS optimization through the standard training/validation
hooks, plus `get_logger`/`get_trainer`/`run_trainer`, the three calls every reconstruction
script in scripts/particle_tracking/ uses to actually run one.
"""

import math
import os

import torch
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ctrex.utils import filetools as ft, ct_setup as cs
from ctrex.optimization import torchtools as tt, samplers
from ctrex.utils.datasets import CTDataset
from ctrex.optimization.samplers import SinogramSampler


class CTReconstruction(LightningModule):
    """ The algorithmic optimizer with ordered subsets updates of the volumes """
    def __init__(self, sinogram: CTDataset, ct_sample: cs.CTSimulation, sampler: SinogramSampler,
                 gt_sample = None, default_lr = 0.54321, data_loss_func = torch.nn.MSELoss(reduction = 'sum')):
        """
        Args:
            sinogram: measured sinogram data being reconstructed against.
            ct_sample: the differentiable forward model (`CTSimulation`) whose parameters
                are being optimized.
            sampler: decides how projections are split into ordered subsets.
            gt_sample: optional ground-truth sample, used only to compute validation metrics
                (e.g. in simulated-data experiments); not used for training.
            default_lr: fallback learning rate for parameters that don't specify their own
                (see `OptimModule.params_to_optimize`).
            data_loss_func: loss comparing measured vs. simulated sinogram samples.
        """
        super().__init__()
        self.sinogram = sinogram
        self.ct_sample = ct_sample
        self.gt_sample = gt_sample
        self.sampler = sampler
        self.validation_projections = self.get_validation_projections(sinogram)
        self.default_lr = default_lr
        self.data_loss_func = data_loss_func
        self.num_optimizers = 1
        self.iterations = 1

        # variables during optimization
        self.current_sampled_projections = None
        self.iteration_loss = 0.
        self.automatic_optimization = False  # LightningModule using manual_backward()
        self.to(self.ct_sample.device)  # Trainer will send it to accelerator device again, which may be cpu!

        self.training_freeze = False
        self.epoch_was_frozen = False  # set during training_step, read by Scheduler to trigger explicit annealing

    def total_steps(self):
        """Total number of OS training steps across all `iterations` epochs - the value
        Lightning's `Trainer` is configured with as `max_steps`."""
        return int(self.iterations * self.sampler.num_subsets * self.num_optimizers)

    @staticmethod
    def get_validation_projections(sinogram):
        """Pick 36 projection angles, evenly spaced across the full sinogram, to use as a
        fixed validation subset (independent of the training OS sampler)."""
        validation_projections = torch.linspace(0, sinogram.num_projections - 1, 36, dtype=torch.int,
                                                device = sinogram.device)  # projection_range
        return validation_projections

    def configure_optimizers(self):
        """Lightning hook: build a single Adam optimizer over the sample model's per-parameter
        learning-rate groups (see `OptimModule.params_to_optimize`)."""
        param_groups = self.ct_sample.params_to_optimize(
            default_lr=self.default_lr
        )
        optimizer = torch.optim.Adam(param_groups)
        # optimizer = torch.optim.SGD(param_groups)
        return [optimizer]

    def collect_learning_rate_hparams(self) -> dict:
        """Collect each optimizer param group's learning rate, keyed as `learning_rate.<name>`,
        for logging as hyperparameters."""
        hparams = {}

        # Trainer + optimizer are available at on_fit_start
        for optimizer in self.trainer.optimizers:
            for i, group in enumerate(optimizer.param_groups):
                name = group.get("name", f"group_{i}")
                lr = group.get("lr")

                hparams[f"learning_rate.{name}"] = lr

        return hparams

    def on_fit_start(self):
        """Lightning hook: enable synchronous CUDA error reporting, and register loss
        weights and learning rates as hyperparameters (for TensorBoard's hparams view)."""
        # https://discuss.pytorch.org/t/how-to-fix-cuda-error-device-side-assert-triggered-error/137553/11
        # torch.autograd.set_detect_anomaly(True)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        # Loss weights (module-level)
        loss_weight_hparams = self.ct_sample.collect_loss_weight_hparams()

        # Learning rates (optimizer-level)
        lr_hparams = self.collect_learning_rate_hparams()
        for name, parameter in self.ct_sample.named_parameters():
            if not parameter.requires_grad:
                print(f"Params grad disabled for parameter {name}")

        # Register all as hyperparameters
        # iterations = math.ceil(self.trainer.max_steps / self.sampler.num_subsets / self.num_optimizers)
        self.hparams.update(loss_weight_hparams)
        self.hparams.update(lr_hparams)
        self.hparams.update({'subset_size': int(self.sampler.subset_size)})
        self.hparams.update({'iterations': int(self.iterations)})


    def on_train_start(self):
        """Lightning hook: free cached CUDA memory, ensure the sample model is on its target
        device, and pre-load the validation projections so they stay resident in memory."""
        torch.cuda.empty_cache()
        self.ct_sample.to(device = self.ct_sample.device)
        # preload projections for visualization first so they remain in memory
        _ = self.sinogram[:, self.validation_projections]
        del _

    def on_train_epoch_start(self):
        """Lightning hook: reset per-epoch bookkeeping (accumulated loss, and whether this
        epoch turns out to have been frozen - see `training_step`)."""
        self.iteration_loss = 0.
        self.epoch_was_frozen = False

    def on_train_epoch_end(self):
        pass

    def training_step(self, batch, batch_idx):
        """
        1 epoch is a full iteration over the projections, observed by the OS sampler.
        1 training step is a single OS update.
        Since the OS sampler is part of the algorithmic domain, let's not make it part of the Dataloader.
        The subset size may be adjusted in a later stage to adapt to the state of the reconstruction.
        """
        # initialize an empty sinogram and compute forward projection for the next ordered subset
        sampled_projections = self.current_sampled_projections
        sinogram_sample = self.sinogram[:, sampled_projections].to(self.ct_sample.device)
        with torch.no_grad():
            simulated_sinogram_sample = torch.zeros_like(sinogram_sample)
        if getattr(self, 'training_freeze', False):  # allow callbacks to disable training temporarily
            with torch.no_grad():
                simulated_sinogram_sample = self.ct_sample.forward(simulated_sinogram_sample, sampled_projections)
            # Don't log a fake epoch/data_loss here (used to log 0, which silently forced
            # ReduceLROnPlateau to anneal since a real loss can never beat it - see Scheduler,
            # which now checks epoch_was_frozen and anneals explicitly instead).
            self.epoch_was_frozen = True
            return {'simulated_sinogram_sample': simulated_sinogram_sample,'sampled_projections': sampled_projections,}

        simulated_sinogram_sample = self.ct_sample.forward(simulated_sinogram_sample, sampled_projections)

        # compute data loss and regularization loss
        loss_dict = self.collect_losses(sinogram_sample, simulated_sinogram_sample, 'train/')

        # Manual backward because parameter graph is not stable across steps (can add particles)
        self.manual_backward(loss_dict['train/total_loss'])
        optimizers = self.optimizers()
        optimizers = [optimizers] if not isinstance(optimizers, list) else optimizers
        for opt in optimizers:  # configure_optimizers should return an iterable
            opt.step()
            opt.zero_grad()

        return {**{'simulated_sinogram_sample': simulated_sinogram_sample,
                   'sampled_projections': sampled_projections},
                **loss_dict}

    def on_train_batch_start(self, batch, batch_idx):
        """Lightning hook: draw the next ordered subset of projections from `self.sampler`
        before the upcoming `training_step`, and print progress."""
        # batch is meaningless: OS logic is part of the reconstructor, not the dataloader
        # the sinogram is part of the CTReconstruction class
        it = self.trainer.current_epoch
        self.current_sampled_projections = self.sampler.next_projections().to(self.sinogram.device)
        # iterations = math.ceil(self.trainer.max_steps / self.sampler.num_subsets / self.num_optimizers)
        ft.printcounting(f"iteration {it}/{self.iterations} - subiteration {batch_idx}/{self.sampler.num_subsets} ",
                         batch_idx, self.total_steps())

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Lightning hook: clamp sample-model parameters back into valid ranges after the
        optimizer step, and accumulate/log the running data loss for the epoch. No-ops if
        training was frozen for this batch (see `training_freeze` in `training_step`)."""
        self.ct_sample.clamp_params()
        if outputs is None or 'train/data_loss' not in outputs: return
        loss_dict = outputs
        data_loss = loss_dict['train/data_loss'].detach()
        # accumulate data_loss over a full iteration
        self.iteration_loss += loss_dict['train/data_loss'].detach() / self.sampler.num_subsets
        self.log('epoch/data_loss', data_loss, on_step=False, on_epoch=True, reduce_fx='mean') #reduce_fx will accumulate


    def on_validation_start(self):
        """Lightning hook: ensure the sample model is on its target device, and warn about
        any parameter Lightning left uninitialized."""
        # Lightning moves model
        self.ct_sample.to(device = self.ct_sample.device)

        for name, param in self.named_parameters():
            if isinstance(param, torch.nn.UninitializedParameter):
                print("UNINITIALIZED PARAM:", name, param)
        pass

    @torch.no_grad()
    def validation_step(self, batch, subiteration):
        """Lightning hook: forward-project the fixed `validation_projections` (not the
        training OS subsets), and compute losses plus ground-truth-comparison metrics
        (if `gt_sample` is set)."""
        sinogram, gt_sample = self.sinogram, self.gt_sample
        sampled_projections = self.validation_projections.to(self.ct_sample.device)

        sinogram_sample = sinogram[:, sampled_projections].to(self.ct_sample.device)
        # initialize an empty sinogram and compute forward projection
        simulated_sinogram_sample = torch.zeros_like(sinogram_sample)
        simulated_sinogram_sample = self.ct_sample(simulated_sinogram_sample, sampled_projections)

        # compute data loss and regularization loss
        metrics_dict = self.collect_losses(sinogram_sample, simulated_sinogram_sample, prefix = 'val/')
        metrics_dict['val/total_loss'] = sum(metrics_dict.values())
        metrics_dict.update(self.collect_metrics(gt_sample, 'val/'))

        return simulated_sinogram_sample, sampled_projections, metrics_dict


    def collect_losses(self, sinogram_sample, simulated_sinogram_sample, prefix =''):
        """ Parent collects all losses from all children. """
        loss_dict = {}

        # Data fidelity
        data_loss = self.data_loss_func(sinogram_sample, simulated_sinogram_sample)
        loss_dict[prefix + "data_loss"] = data_loss

        # Regularization loss computed by all registered_losses
        reg_loss_dict = self.ct_sample.collect_losses(None, prefix)
        loss_dict.update(reg_loss_dict)

        # aggregate all separate losses
        loss_dict['train/total_loss'] = sum(loss_dict.values())

        # Invariant loss to compare over reconstructions with different loss choices
        # not part of total_loss
        diff = (simulated_sinogram_sample - sinogram_sample) ** 2
        diff[torch.isinf(sinogram_sample)] = 0
        mse_nan = torch.nanmean(diff)
        loss_dict[prefix + 'mse_nan'] = mse_nan
        return loss_dict

    def collect_metrics(self, gt_sample, prefix =''):
        """Delegate to the sample model to compute ground-truth-comparison metrics."""
        metrics_dict = self.ct_sample.collect_metrics(gt_sample, prefix)
        return metrics_dict


    def get_train_dataloader(self):
        """Build the training dataloader: one epoch = one full OS sweep (see `SinogramSampler`)."""
        return self.sampler.get_os_epoch_dataloader()  # One epoch = one full sweep of OS training steps

    @staticmethod
    def get_val_dataloader():
        """Build the validation dataloader: one epoch = one call to `validation_step`."""
        return DataLoader(samplers.SingleStepDataset(), batch_size=None, num_workers=0)  # One epoch = one validation step


def get_logger(ct_dataset: CTDataset):
    """Build the TensorBoard logger for a reconstruction run: the run is named after the
    scan (`ct_dataset.scan_name`, falling back to the scan folder's basename), and versioned
    by the current git branch/commit so different code states don't overwrite each other's logs."""
    # Set up visualizer
    assert hasattr(ct_dataset, 'scan_name')
    assert hasattr(ct_dataset, 'scan_folder')
    scan_name = ct_dataset.scan_name
    scan_name = os.path.basename(ct_dataset.scan_folder) if scan_name is None else scan_name

    # Code version
    active_branch_name = ft.get_active_branch_name()
    version_name = ft.get_version_name()  # Retrieve code state based on f"{date_time}_{short_sha}"
    print(active_branch_name, version_name, scan_name, ct_dataset.scan_folder)

    logger = tt.TBLogger(
        save_dir = os.path.join("..", "runs"),
        name = scan_name,  # Reconstruction model
        version = version_name,  # Tweaks to that reconstruction model
        sub_dir = None,
        log_graph = True,
        default_hp_metric = False
    )
    return logger


def get_trainer(ct_reconstruction: CTReconstruction, iterations, num_validations,
                logger, callbacks = (), check_val_every_n_epoch=1):
    """
    Get a Lightning Trainer that orchestrates the CT reconstruction and callbacks
    """
    # Control number of steps instead of epochs to allow partial iterations
    ct_reconstruction.iterations = iterations
    max_steps = ct_reconstruction.total_steps()
    val_check_interval = max(1,max_steps // num_validations)
    trainer = Trainer(accelerator='gpu',
                      devices=1,
                      num_sanity_val_steps=1,
                      max_steps=max_steps,
                      max_epochs = iterations,
                      val_check_interval=val_check_interval,
                      logger = logger,
                      log_every_n_steps=1,
                      check_val_every_n_epoch=check_val_every_n_epoch,
                      callbacks=list(callbacks),
                      enable_checkpointing=False)
    return trainer

def run_trainer(trainer: Trainer, ct_reconstruction: CTReconstruction):
    """
    Let the CT reconstruction algorithm create simple dataloaders
    """
    # The dataloaders in this reconstruction are just responsible for defining epochs and subset steps
    # One epoch = one iteration runs over a full dataset, one step is one subset update
    dl_train = ct_reconstruction.get_train_dataloader()
    dl_validation = ct_reconstruction.get_val_dataloader()
    trainer.fit(ct_reconstruction, dl_train, dl_validation)