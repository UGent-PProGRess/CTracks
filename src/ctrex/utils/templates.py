"""
Convenience methods to get CT simulators with some default settings

"""

import copy
import torch

from ctrex.optimization import samplers
from ctrex.utils import datasets as dl, ct_setup as cs
from ctrex.sample_description import track_models as tm, shape_models as sm
from ctrex.sample_description.PoreMaskFunctions import PoreMaskPlain
from ctrex.optimization.reconstruction import get_logger, CTReconstruction, get_trainer, run_trainer
from ctrex.sample_description.sample_component import SampleComponent
from ctrex.projectors import SequentialCTModule, SIRTCTSimulation, ArrayIntersectionCTSimulation
from ctrex.optimization.visualize_results import VisualizeResults


def add_static_matrix_projector(ct_simulation: cs.CTSimulation, voxel_scale = 1, roi = None,
                                vol_limits = None, num_components = 1, device = None) -> None:
    """Add a 'static_matrix' projector to `ct_simulation`: a non-moving voxel-array sample
    component (a `VolumeArray` shape model on a fixed `StaticTrack`) covering the volume of
    interest derived from `roi`, at `voxel_scale` resolution and attenuation clamped to
    `vol_limits`. This is the reconstruction target for the static/background part of the
    sample (e.g. the porous matrix around moving particles)."""
    device = device or ct_simulation.device
    vol_shape = ct_simulation.trajectory.volume.vol_shape
    voi, voi_shape, voi_centre = ct_simulation.trajectory.roi_to_voi(roi)
    voi_shape = (int(voi_shape[0] // voxel_scale), int(voi_shape[1] // voxel_scale), int(voi_shape[2] // voxel_scale))

    pore_mask = PoreMaskPlain(vol_shape, device, voi = voi)
    component_volume = SampleComponent(
        tm.StaticTrack(2, device, vol_shape, learning_rate = 0,
                       bounds = [0] * 6,
                       # bounds = [half_size for dim in range(3) for half_size in (vol_shape[2-dim] / 2,)*2],
                       pore_mask = pore_mask, loss_weight=0, mask_confines=False),
        sm.VolumeArray(3, device, voi_shape, voxel_scale, attenuation_range=vol_limits,
                       learning_rate=0.6),
    )

    ct_simulation.add_projector('static_matrix', SIRTCTSimulation(component_volume, sampling_rate=1))
    component_volume.init_component(num_components)  # 1 volume
    with torch.no_grad():
        # component_volume.track_model.control_points.data = roi_centre.unsqueeze(0).unsqueeze(0)
        component_volume.track_model.control_points.data = voi_centre.unsqueeze(0).unsqueeze(0)

def add_flow_cell_projector(ct_simulation: cs.CTSimulation, device = None) -> None:
    """Add a 'flow_cell' projector to `ct_simulation`: a fixed, non-learnable set of nested
    cylinders (an inner sample bore, a confining wall, and an outer PEEK housing, with
    attenuations set accordingly) representing the physical flow-cell holding the sample, so
    the reconstruction can account for its contribution to the sinogram without trying to fit
    it as sample."""
    device = device or ct_simulation.device
    voxel_size = ct_simulation.trajectory.voxel_size
    vol_shape = ct_simulation.trajectory.volume.vol_shape
    voi, voi_shape, voi_centre = ct_simulation.trajectory.roi_to_voi(None)

    component_flow_cell = SampleComponent(
        tm.StaticTrack(2, device, vol_shape, mask_confines=False),
        sm.CylinderArray(3, device, (0, 3.0), radii=torch.tensor([8, 12, 14]) / 2,
                         tilt_std=0, bounds=10 / 2 / voxel_size,
                         tilt_range=(-10, 10))
    )
    component_flow_cell.track_model.learning_rate = 0.3
    # set learning rate for attenuations on the higher side so static matrix does not try to take over attenuation
    component_flow_cell.shape_model.learning_rates = {'attenuations': 1e-1, 'dx_top': 1e-1, 'dy_top': 1e-1}
    component_flow_cell.init_component(1, tm.init_zeros, sm.init_zeros)
    flow_cell_projector = ArrayIntersectionCTSimulation(component_flow_cell)
    ct_simulation.add_projector('flow_cell', flow_cell_projector)

    with torch.no_grad():
        component_flow_cell.track_model.control_points.data = voi_centre.unsqueeze(0).unsqueeze(0)
        component_flow_cell.shape_model.attenuations[:,0] = 0.0  # sample inside - don't interfere
        component_flow_cell.shape_model.attenuations[:,1] = 0.3 # confining
        component_flow_cell.shape_model.attenuations[:,2] = 0.8  # peek


def reconstruct_dataset(ct_dataset: dl.CTDataset, ct_sample: cs.CTSimulation, recon_name,
                        iterations = 1, subset_size = 1, callbacks = (), num_frames = 3,
                        reconstruct = True, learning_rate = 0.5, gt_sample = None,
                        data_loss_func = torch.nn.MSELoss(reduction = 'sum'), projection_indices = None):
    """Assemble a CTReconstruction over `ct_dataset`/`ct_sample` (with an ordered-subset
    sampler over `projection_indices`, defaulting to all of them) and a lightning trainer
    around it, then run it unless `reconstruct` is False (in which case the trainer/
    reconstructor are returned unrun, e.g. for inspection or manual stepping).

    Returns:
        trainer, reconstructor: the configured (and possibly already-run) trainer and
            CTReconstruction instances.
    """
    logger = get_logger(ct_dataset)
    visualizer = VisualizeResults(ct_dataset, recon_name)
    callbacks = (visualizer,) + callbacks
    projection_indices = projection_indices or range(ct_dataset.num_projections)

    sampler = samplers.OrderedSubsetSampler(projection_indices, subset_size=subset_size)
    reconstructor = CTReconstruction(ct_dataset, ct_sample, sampler, gt_sample=gt_sample, default_lr=learning_rate,
                                     data_loss_func=data_loss_func)
    trainer = get_trainer(reconstructor, iterations, num_frames, logger, callbacks)
    if reconstruct:
        run_trainer(trainer, reconstructor)
    return trainer, reconstructor

def reconstruct_dynamic_dataset(ct_dataset: dl.CTDataset, ct_sample: cs.CTSimulation, recon_name,
                                init_sample = None, num_proj = None, proj_shift = None, **kwargs):
    """Reconstruct `ct_dataset` in a sliding window of `num_proj` projections (defaulting to
    one rotation's worth), stepping by `proj_shift` (defaulting to `num_proj`, i.e.
    non-overlapping windows) and re-initializing `ct_sample` close to `init_sample` (a deep
    copy of `ct_sample` if not given) before each window's `reconstruct_dataset` call - useful
    for reconstructing a time-varying sample frame-by-frame. NOTE: the only reference to this
    function found in scripts/ (recon_multiphase.py) is commented out, so treat as unverified
    against the current `reconstruct_dataset`/`CTSimulation` signatures until re-checked."""
    if init_sample is None:
        init_sample = copy.deepcopy(ct_sample)
    num_proj = int(ct_sample.trajectory.proj_per_rot) if num_proj is None else num_proj
    proj_shift = num_proj if proj_shift is None else proj_shift
    for start_proj in range(0, int(ct_dataset.num_projections), proj_shift):
        recon_name_time = f'{recon_name}_{start_proj:06d}_{start_proj+num_proj:06d}'
        print(recon_name_time)
        ct_sample.init_close(init_sample, 0, 0)
        reconstruct_dataset(ct_dataset, ct_sample, recon_name_time, **kwargs,
                            projection_indices=range(start_proj, start_proj + num_proj))
