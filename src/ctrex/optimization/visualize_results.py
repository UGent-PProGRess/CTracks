"""Plotting and Lightning-callback helpers for visualizing reconstruction progress.

Provides the ``VisualizeResults`` callback (logs losses/geometric parameters and writes
sinogram-comparison and volume-slice figures to TensorBoard during training/validation),
the individual comparison-plot functions it can use (``plot_overlap_sinogram``,
``plot_difference_sinogram``, ``plot_ground_truth_sinogram``), and lower-level helpers for
slicing/plotting attenuation volumes.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import os
import pytorch_lightning as pl
from lightning import Trainer, Callback

from ctrex.optimization.reconstruction import CTReconstruction
from ctrex.utils import filetools as ft
from ctrex.optimization import torchtools as tt
from ctrex.sample_description import shape_models as sm
from ctrex.utils.datasets import CTDataset

def remove_axes(ax, remove_ticklines = True):
    """Strip axis labels/tick labels (and, if ``remove_ticklines``, the tick marks themselves)
    from a 2D or 3D matplotlib axis, for a cleaner figure."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if remove_ticklines:
        ax.set_xticks([])
        ax.set_yticks([])
    try:
        ax.set_zlabel("")
        ax.set_zticklabels([])
        if remove_ticklines:
            ax.set_zticks([])
    except AttributeError:
        pass


def plot_overlap_sinogram(sinogram, reconstructed_sinogram, ax, max_scaling = None, ylabel =r"$\theta~(°)$", cmap ="white",
                          y_axis = None, clean = False, **_plot_params):
    """Plot the ground-truth and reconstructed sinograms overlaid on ``ax`` as two color channels
    (e.g. magenta vs. cyan for ``cmap='white'``), so agreement shows as white/black and mismatch
    shows as color.

    Args:
        max_scaling: optional single-element list used as a mutable "output" - if its value is
            None, it is filled in with the scaling used here so later calls (e.g. other subplots)
            can reuse the same intensity scale.
        y_axis: optional physical y-axis values (e.g. angles) to label the ticks with, instead of
            raw pixel indices.
        clean: if True, strip axis decorations via ``remove_axes`` for a publication-style figure.
    """
    cmap = {'white':np.array([[-255, -127, 0], [0, -127, -255], [255,255,255]]),
            'black':np.array([[0, 0, 255], [0, 255, 0], [0,0,0]])}.get(cmap, "white")[:,np.newaxis, np.newaxis]
    max_val = reconstructed_sinogram.max()
    if max_scaling is not None:  # [None] or [some value]
        if max_scaling[0] is None:
            max_scaling[0] = max_val  # update value in modifiable list
        max_val = max_scaling[0]
    sino1 = np.tile(sinogram[..., np.newaxis], [1, 1, 3]) / sinogram.max() * cmap[0]
    sino2 = np.tile(reconstructed_sinogram[..., np.newaxis], [1, 1, 3]) / max_val * cmap[1]
    overlap = (cmap[2] + sino1 + sino2).astype(np.uint8)
    ax.imshow(overlap, aspect ='auto', interpolation = None, origin ='upper',
              extent = [-0.5, sino1.shape[1] + 0.5, y_axis[-1] + 0.5, y_axis[0] - 0.5] if y_axis is not None else None)
    ax.set_xlabel('u')
    ax.set_ylabel(ylabel)
    if y_axis is not None:
        ax.set_yticks(np.append(y_axis[::len(y_axis) // 4], y_axis[-1:]))  # including final value.
    if clean:
        remove_axes(ax)
        plt.tight_layout()


def plot_difference_sinogram(sinogram, reconstructed_sinogram, ax, max_scaling = None, ylabel =r"$\theta~(°)$",
                             cmap ="seismic", y_axis = None, clean = False, **_plot_params):
    """Plot ``sinogram - reconstructed_sinogram`` on ``ax`` with a diverging colormap centered
    at zero, plus a colorbar.

    Args:
        max_scaling: optional single-element list used as a mutable "output" - if its value is
            None, it is filled in with the symmetric color scale used here so later calls can
            reuse it.
        y_axis: optional physical y-axis values (e.g. angles) to label the ticks with, instead of
            raw pixel indices.
        clean: if True, strip axis decorations via ``remove_axes`` for a publication-style figure.
    """
    difference = sinogram - reconstructed_sinogram
    max_diff = max(np.abs(difference.max()), np.abs(difference.min()))
    if max_scaling is not None:  # [None] or [some value]
        if max_scaling[0] is None:
            max_scaling[0] = max_diff  # update value in modifiable list
        max_diff = max_scaling[0]
    ax.imshow(difference, aspect ='auto', interpolation = None, origin ='upper',
              vmin = - max_diff, vmax = max_diff, cmap = cmap,
              extent = [-0.5, difference.shape[1] + 0.5, y_axis[-1] + 0.5, y_axis[0] - 0.5] if y_axis is not None else None)
    norm = plt.Normalize(-max_diff, max_diff)
    scm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ax.figure.colorbar(scm, ax=ax, orientation='vertical')
    ax.set_xlabel('u')
    ax.set_ylabel(ylabel)
    if y_axis is not None:
        ax.set_yticks(np.append(y_axis[::len(y_axis) // 4], y_axis[-1:]))  # including final value.
    if clean:
        remove_axes(ax)
        plt.tight_layout()


def plot_ground_truth_sinogram(sinogram, _reconstructed_sinogram, ax, max_scaling = None, ylabel =r"$\theta~(°)$",
                               cmap ="seismic", y_axis = None, clean = False, **_plot_params):
    """Plot just the ground-truth ``sinogram`` on ``ax`` (ignoring ``_reconstructed_sinogram``),
    with a colorbar. Kept signature-compatible with ``plot_overlap_sinogram``/
    ``plot_difference_sinogram`` so it can be swapped in as ``VisualizeResults.plot_comparison``.

    Args:
        max_scaling: optional single-element list used as a mutable "output" - if its value is
            None, it is filled in with the max value used to scale the color range here.
        y_axis: optional physical y-axis values (e.g. angles) to label the ticks with, instead of
            raw pixel indices.
        clean: if True, strip axis decorations via ``remove_axes`` for a publication-style figure.
    """
    max_val = sinogram.max()
    if max_scaling is not None:  # [None] or [some value]
        if max_scaling[0] is None:
            max_scaling[0] = max_val  # update value in modifiable list
        max_val = max_scaling[0]
    ax.imshow(sinogram, aspect ='auto', interpolation = None, origin ='upper',
              vmin = 0, vmax = max_val, cmap = cmap,
              extent = [-0.5, sinogram.shape[1] + 0.5, y_axis[-1] + 0.5, y_axis[0] - 0.5] if y_axis is not None else None)
    norm = plt.Normalize(0, max_val)
    scm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ax.figure.colorbar(scm, ax=ax, orientation='vertical')
    ax.set_xlabel('u')
    ax.set_ylabel(ylabel)
    if y_axis is not None:
        ax.set_yticks(np.append(y_axis[::len(y_axis) // 4], y_axis[-1:]))  # including final value.
    if clean:
        remove_axes(ax)
        plt.tight_layout()


def slice_attenuation(ct_sample, dim, start_projection = ()):
    """Sample a single central 2D slice of ``ct_sample``'s attenuation volume perpendicular to
    axis ``dim`` (0=z, 1=y, 2=x), at the given ``start_projection`` (default: initial state).

    Returns:
        attenuations: the 2D slice (numpy array).
        extent: (left, right, bottom, top) pixel-coordinate bounds, for use with ``imshow``.
        sampled_shape_params: the shape-model parameters sampled at this slice.
        limits: the attenuation value range used for color-scaling.
    """
    dims = [(2 - di) for di in range(3) if di != dim][::-1]  # zyx -> yx -> xy
    device = ct_sample.device  # (depth z, height y, width x)
    vol_shape = ct_sample.trajectory.volume.shape
    extent = ct_sample.extent(start_projection)  # xyz
    coordinates_grid = torch.meshgrid([(torch.arange(int(extent[0][2 - di]), int(extent[1][2 - di]),
                                                     dtype=torch.int32, device=device)
                                        if di != dim
                                        else torch.tensor([vol_shape[dim] // 2], dtype=torch.int32, device=device))
                                       for di in range(3)], indexing='ij')
    coordinates_grid = torch.stack(coordinates_grid[::-1], dim=-1)  # (nz, ny, nx, 3 -> xyz)
    # attenuations at start time
    attenuations, sampled_shape_params, limits = ct_sample.sample_attenuations(
        coordinates_grid, start_projection)
    attenuations = attenuations.mean(-1)  # average over time domain (of a single projection)
    cg = coordinates_grid.squeeze(dim)
    # extent is left, right, bottom, top
    extent = [ext.cpu().numpy() for i, di in enumerate(dims) for ext in
              (cg[..., di].min(), cg[..., di].max())[::1 - 2 * i]]
    attenuations = attenuations.squeeze(dim).detach().cpu().numpy()
    return attenuations, extent, sampled_shape_params, limits

def imshow_colorbar(ax, attenuations, limits, cmap ='gray', **kwargs):
    """Show ``attenuations`` on ``ax`` with a colorbar, scaled to ``limits`` and with
    invalid/masked values drawn white."""
    cmap = mpl.colormaps.get_cmap(cmap)
    cmap.set_bad(color = 'white')
    img = ax.imshow(attenuations, aspect='auto', interpolation='nearest', origin='upper',
                    vmin=limits[0], vmax=limits[1], cmap=cmap, **kwargs)
    norm = plt.Normalize(*img.get_clim())
    scm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ax.figure.colorbar(scm, ax=ax, orientation='vertical')


def clear_folder(folder_path):
    """Create ``folder_path`` if missing, and delete every file directly inside it.

    NOTE: not currently called anywhere.
    """
    os.makedirs(folder_path, exist_ok=True)
    # delete everything in animation folder
    print(f"Clearing existing animation in folder {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.unlink(file_path)


class VisualizeResults(Callback):
    """Lightning callback that logs reconstruction progress to TensorBoard: per-parameter
    scalars/histograms after each backward pass (``on_after_backward``), losses and a
    projection-error plot after each training batch (``on_train_batch_end``), and sinogram
    comparison + volume-slice figures plus hyperparameter/metric logging after each validation
    batch (``on_validation_batch_end``, via ``plot_recon_results``). On training end, also
    writes the reconstructed volume to a TIFF file if a ``static_matrix`` shape model is present
    (``on_train_end``)."""
    def __init__(self,
                 ct_dataset: CTDataset,
                 recon_name,

                 plot_comparison = [plot_overlap_sinogram, plot_difference_sinogram, plot_ground_truth_sinogram][1],
                 cmap = ['white', 'seismic'][1],
                 max_scaling = None,
                 clean = True,
                 plot_3d = False,
                 truth_sample = None,
                 init_sample = None):
        """
        Args:
            recon_name: base name for output files/figures; a suffix identifying the scan and ROI
                is appended automatically.
            plot_comparison: which sinogram-comparison plot function to use (default:
                ``plot_difference_sinogram``).
            cmap: colormap passed through to ``plot_comparison`` (default: 'seismic').
            max_scaling: fixed color-scale value to reuse across figures/subplots instead of
                recomputing it per-plot; wrapped in a list internally to make it mutable.
            plot_3d: currently unused by this class - kept for compatibility with callers/subclasses.
            truth_sample, init_sample: currently unused by this class - kept for compatibility with
                callers/subclasses.
        """
        super().__init__()
        self.scan_folder = ct_dataset.scan_folder
        self.scan_name = ct_dataset.scan_name
        recon_suffix = f'_{os.path.split(ct_dataset.scan_name)[-1]}_{ct_dataset.roi[0].start:04d}_{ct_dataset.roi[0].stop:04d}'
        self.recon_name = recon_name + recon_suffix
        os.makedirs(self.scan_folder, exist_ok=True)

        self.plot_comparison = plot_comparison
        self.cmap = cmap
        self.max_scaling = [max_scaling]  # list to make this mutable
        self.clean = clean
        self.plot_3d = plot_3d
        self.truth_sample = truth_sample
        self.init_sample = init_sample
        self.projection_errors = []

    def on_fit_start(self, trainer: Trainer, ct_recon: CTReconstruction) -> None:
        """Reset the projection-error history at the start of training."""
        self.projection_errors = []

    def on_after_backward(self, trainer: pl.Trainer, ct_recon: pl.LightningModule) -> None:  # save geometric parameters
        """Log every learnable parameter/buffer to TensorBoard: as a scalar if it has a single
        element, skipped if it doesn't require grad (e.g. rotation/translation axes), otherwise
        as a histogram (plus its gradient's histogram) every 100 steps."""
        # noinspection PyUnresolvedReferences
        writer = trainer.logger.experiment  # type: tt.SummaryWriter
        trainer_step = trainer.fit_loop.total_batch_idx
        state_dict = ct_recon.ct_sample.small_state_dict(numel = 1e100)  # parameters + buffers
        for name, param in state_dict.items():  # type: str, torch.Tensor
            if param.numel() == 1:
                writer.add_scalar(name, param.detach().cpu().numpy().item(), global_step=trainer_step)
            elif not param.requires_grad:  # rotation axis, translation axis etc.
                pass
            else:
                if trainer_step % 100 != 0:  # histograms may be expensive
                    continue
                writer.add_histogram(name, param.detach().cpu().numpy(), global_step=trainer_step)
                if param.grad is not None:
                    try:
                        writer.add_histogram(name + '_grad', param.grad.detach().cpu().numpy(), global_step=trainer_step)
                    except ValueError:  # histogram empty for some reason
                        pass


    def on_train_batch_end(self, trainer: Trainer, ct_recon: CTReconstruction, outputs: dict, batch, batch_idx) -> None:
        """
        Store losses to tensorboard and make figures

        Logs each loss in ``outputs`` as a scalar, tracks the mean per-projection sinogram error
        (converted to 1/cm) for the batch's sampled projections, and writes a rolling
        projection-error plot (see ``plot_projection_errors``).
        """
        # noinspection PyUnresolvedReferences
        writer = trainer.logger.experiment  # type: tt.SummaryWriter
        trainer_step = trainer.fit_loop.total_batch_idx
        simulated_sinogram_sample = outputs.pop('simulated_sinogram_sample')
        sampled_projections = outputs.pop('sampled_projections')
        loss_dict = outputs

        for name, val in loss_dict.items():
            writer.add_scalar(name, val.item(), global_step=trainer_step)

        # log projection errors and visualize
        sinogram_sample = tt.tensor_to_numpy(ct_recon.sinogram[:, sampled_projections])  # type: np.array
        simulated_sinogram_sample = tt.tensor_to_numpy(simulated_sinogram_sample)  # type: np.array
        errors = (sinogram_sample - simulated_sinogram_sample).mean(axis=(0, 2))
        errors /= ct_recon.ct_sample.trajectory.voxel_size.item() / 10  # get some value in 1/cm
        self.projection_errors.append((sampled_projections, errors))
        fig_frame = self.plot_projection_errors()
        writer.add_figure('projection errors', fig_frame, global_step=trainer_step)
        plt.close(fig_frame)

        # device = ct_sample.device
        # plot_projections = ct_sample.trajectory.projection_indices(ct_recon.validation_projections)
        # empty_sinogram = ct_sample.empty_sinogram(CTDatasetSparse, mem_projections=self.mem_projections,
        #                                           device=device)
        # views = ct_sample.trajectory(plot_projections)
        # writer.add_graph(ct_sample.projectors, (empty_sinogram[:, plot_projections], views, plot_projections))
        # state_dict_file = os.path.join(self.scan_folder, f'{self.recon_name}_{version_name}.pth')
        # torch.save(ct_recon.ct_sample.small_state_dict(), state_dict_file)
        writer.flush()

    @torch.no_grad()
    def on_validation_batch_end(self, trainer: Trainer, ct_recon: CTReconstruction, outputs,
                                batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Log hyperparameters and validation metrics to TensorBoard, and write a reconstruction
        comparison figure (see ``plot_recon_results``)."""
        # noinspection PyTypeChecker
        logger = trainer.logger  # type: tt.TBLogger
        writer = logger.experiment  # type: tt.SummaryWriter
        simulated_sinogram_sample, sampled_projections, metrics_dict = outputs
        sinogram_sample = tt.tensor_to_numpy(ct_recon.sinogram[:, sampled_projections])  # type: np.array
        simulated_sinogram_sample = tt.tensor_to_numpy(simulated_sinogram_sample)  # type: np.array
        sampled_projections = tt.tensor_to_numpy(sampled_projections)
        ct_sample = ct_recon.ct_sample
        trainer_step = trainer.fit_loop.total_batch_idx

        # writes scalars of metrics_dict too
        hparam_dict = dict(ct_recon.hparams)
        writer.add_hparams(hparam_dict, metrics_dict, run_name=logger.version, global_step=trainer_step)

        # visualize the outcome of the reconstruction
        fig_frame = self.plot_recon_results(ct_sample, sinogram_sample,simulated_sinogram_sample,sampled_projections)
        writer.add_figure('reconstruction ' + logger.name, fig_frame, global_step=trainer_step)
        plt.close(fig_frame)

        writer.flush()

    def plot_projection_errors(self, len_buffer = 10):
        """Plot the last ``len_buffer`` recorded per-projection error curves (see
        ``self.projection_errors``), fading older curves out and highlighting the most recent one."""
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 1, 1)
        cmap = mpl.colormaps['viridis']

        for step, (projection_indices, projection_error_step) in enumerate(self.projection_errors[-len_buffer:]):
            ax.plot(projection_indices, projection_error_step,
                    c = cmap(step / len_buffer),
                    alpha = 0.2 if step < len_buffer - 1 else 1)
        ax.set_ylabel('projection error (1/cm)')
        ax.set_xlabel('projection index (-)')
        return fig

    def plot_recon_results(self, ct_sample_recon, sinogram, reconstructed_sinogram, sampled_projections):
        """Build a 2x2 figure summarizing the reconstruction: a central-slab and a full-depth
        max-intensity ``self.plot_comparison`` of sinogram vs. reconstruction (top-left/bottom-left),
        and central z- and x-slices through the reconstructed attenuation volume
        (top-right/bottom-right, via ``slice_attenuation``/``imshow_colorbar``)."""

        # Display the original and reconstructed particle positions
        fig = plt.figure(figsize=(12, 6))

        # plot central difference sinogram
        ax = plt.subplot(2, 2, 1)
        height = sinogram.shape[0]
        angles = ct_sample_recon.trajectory.angles[sampled_projections].detach().cpu().numpy()
        self.plot_comparison(np.max(sinogram[height // 8: - height // 8], axis=0),
                        np.max(reconstructed_sinogram[height // 8: - height // 8], axis=0),
                        ax, self.max_scaling, y_axis=angles, cmap = self.cmap)

        # plot aggregate difference projection
        ax = plt.subplot(2, 2, 3)
        self.plot_comparison(np.max(sinogram, axis=1),
                        np.max(reconstructed_sinogram, axis=1),
                        ax, self.max_scaling, ylabel='v', cmap = self.cmap)

        ax = fig.add_subplot(2, 2, 2)
        attenuations, extent, shape_params, limits = slice_attenuation(ct_sample_recon, 0, [0])  # z
        imshow_colorbar(ax, attenuations, limits, extent = extent)

        ax = fig.add_subplot(2, 2, 4)
        attenuations, extent, shape_params, limits = slice_attenuation(ct_sample_recon, 2, [0])  # x
        imshow_colorbar(ax, attenuations, limits, extent = extent)

        return fig

    def on_train_end(self, trainer, ct_recon) -> None:
        """Called when the train ends."""
        # Writes the reconstructed attenuation volume to a TIFF file, if the reconstructed sample
        # has a ``static_matrix`` (matrix/voxel-grid) shape model; does nothing otherwise (e.g. for
        # particle-only reconstructions).
        recon_name = self.recon_name
        if not hasattr(ct_recon.ct_sample, 'static_matrix'):
            return
        static_matrix = ct_recon.ct_sample.static_matrix  # type: sm.VolumeArray
        limits = static_matrix.limits
        reconstructed_volume = static_matrix.attenuations[0].detach().cpu().numpy()
        recon_volume_file = os.path.join(self.scan_folder, f'{recon_name}.tif')
        ft.write_tiff(recon_volume_file, reconstructed_volume, limits, np.uint16)
