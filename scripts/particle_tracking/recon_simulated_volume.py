"""
Reconstruct a static attenuation volume (not particles) from a set of previously
saved simulated projection sets, one output volume per set.

Each "frame" here is an independent, already-existing projection set on disk (e.g.
produced by recon_simulated_linear.py); this script does not track particles
over time, it performs a single-shot SIRT-style volume reconstruction per set and
writes out a reconstructed attenuation volume tiff for each.
"""

import collections
import os
import numpy as np
import torch

from ctrex.optimization import samplers
from ctrex.utils import datasets as dl, filetools as ft, ct_setup as cs
from ctrex.sample_description.sample_component import SampleComponent, tm, sm
from ctrex.sample_description.PoreMaskFunctions import PoreMaskPlain
from ctrex.projectors import *
from ctrex.optimization.reconstruction import CTReconstruction, get_trainer, get_logger, run_trainer

# ---------------------------------------------------------------------------
# Data root - change this if CTracksSubmissionData is moved or mounted under a
# different drive/path (e.g. moving to a new PC).
# ---------------------------------------------------------------------------
DATA_ROOT = "J:\\"

# ---------------------------------------------------------------------------
# Source projection set - select the active one below. proj_fmt names the frame
# subfolder prefix under scan_base (e.g. scan_base + proj_fmt + "0").
# ---------------------------------------------------------------------------
SOURCES = {
    "porevisco_27Xfaster": dict(
        # proj_rot*/proj_recon_rot* inputs live under CTracksSubmissionData's saved_projections\
        # folder; the reconstructions\ output folder lives separately under rdl_results\ (see
        # recon_folder below), not nested under scan_base.
        scan_base=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Simulated\saved_projections"),
        proj_fmt="proj_rot",
    ),
}
ACTIVE_SOURCE = "porevisco_27Xfaster"
source_cfg = SOURCES[ACTIVE_SOURCE]
scan_base = source_cfg["scan_base"]
proj_fmt = source_cfg["proj_fmt"]

recon_folder = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Simulated\rdl_results\reconstructions2")

# ---------------------------------------------------------------------------
# Sinogram / acquisition geometry
# ---------------------------------------------------------------------------
SCAN_GEOMETRY = dict(
    detector_height=500,
    detector_width=400,
    proj_per_rot=851,
    num_rot=1,
    sod=18.3,
    sdd=450,
)
ACQUISITION_PARAMS = dict(
    dimension=3,
    sample_rate=1,
    skew=0,
    tilt=0.0,
    clockwise=-1,
    binning=1,
    pixel_size=0.3,
)
num_projections = SCAN_GEOMETRY["proj_per_rot"] * SCAN_GEOMETRY["num_rot"]

voxel_scale = 1
roi = slice(0, SCAN_GEOMETRY["detector_height"]), slice(0, SCAN_GEOMETRY["detector_width"])
subset_size = 20
mem_projections = num_projections  # during the reconstruction, keep this many projections in (GPU) memory

# ---------------------------------------------------------------------------
# Particle / volume initialisation parameters
# ---------------------------------------------------------------------------
# attenuation range assumed for the reconstructed static volume
VOLUME_ATTENUATION_RANGE = (-0.6, 4.2)

# ---------------------------------------------------------------------------
# Training / run parameters
# ---------------------------------------------------------------------------
TOTAL_FRAMES = 15
TRAINING_PARAMS = dict(
    iterations=1,  # single-shot SIRT-style reconstruction, no need for multiple passes
    # Also used as the volume's shape-model "learning_rate" (see reconstruct_frame): that field
    # doubles as the SIRT relaxation factor, which cancels out mathematically in SIRTProjector's
    # backward pass, so this value only matters as the Adam step size for the attenuations.
    default_lr=0.5,
)

# ---------------------------------------------------------------------------
# Sinogram parameters passed to the CT reconstruction (built from SCAN_GEOMETRY /
# ACQUISITION_PARAMS above)
# ---------------------------------------------------------------------------
sino_params = {
    'width': SCAN_GEOMETRY["detector_width"],
    'height': SCAN_GEOMETRY["detector_height"],
    'angles': np.linspace(0., SCAN_GEOMETRY["num_rot"] * 360., num_projections, endpoint=True, dtype=np.float32),
    'dimension': ACQUISITION_PARAMS["dimension"],
    'centre_of_rotation': SCAN_GEOMETRY["detector_width"] / 2,
    'sample_rate': ACQUISITION_PARAMS["sample_rate"],
    'skew': ACQUISITION_PARAMS["skew"],
    'tilt': ACQUISITION_PARAMS["tilt"],
    'clockwise': ACQUISITION_PARAMS["clockwise"],
    'sod': SCAN_GEOMETRY["sod"],
    'sdd': SCAN_GEOMETRY["sdd"],
    'vertical_centre': SCAN_GEOMETRY["detector_height"] / 2,  # detector height / 2
    'horizontal_centre': SCAN_GEOMETRY["detector_width"] / 2,
    'binning': ACQUISITION_PARAMS["binning"],
    'pixel_size': ACQUISITION_PARAMS["pixel_size"],
    'num_projections': num_projections,
    'roi': (slice(0, SCAN_GEOMETRY["detector_height"]), slice(0, SCAN_GEOMETRY["detector_width"]))
}

# ---------------------------------------------------------------------------
# System / technical setup (not user-facing - leave alone unless you know what
# you're doing)
# ---------------------------------------------------------------------------
np.random.seed(4)  # For reproducibility
torch.manual_seed(0)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.version.__version__)
print("device =", device, type(device))


def reconstruct_frame(frame_index, sino_params):
    """Reconstruct a single static attenuation volume from the on-disk projection set
    for one frame, and write the result to a tiff under recon_folder.

    Args:
        frame_index: index of the frame/projection set to reconstruct.
        sino_params: base sinogram/geometry parameters (mutated in place with the
            frame's actual num_projections/roi, matching the original script's behaviour).
    """
    scan_folder = os.path.join(scan_base, proj_fmt + str(frame_index))
    recon_volume_file = os.path.join(recon_folder, f'frame_{frame_index}.tif')
    # Define the acquired CT dataset
    sinogram = dl.CTDatasetOnDisk((scan_folder, f'proj_%s.tif'), slice(0, None, 1), device,
                                  mem_projections=None,  # all
                                  flat_field=None, dark_field=None,
                                  roi=roi, fixed_memory=mem_projections - subset_size)  # roi gets truncated to full width

    ct_sample_scan = cs.CTScan(sinogram, sino_params)  # computes trajectory with initialized sino_params

    # Define a pore mask
    vol_shape = ct_sample_scan.trajectory.volume.vol_shape
    roi_shape = (sinogram.rheight // voxel_scale,) + (sinogram.rwidth // voxel_scale,) * 2
    pore_mask = PoreMaskPlain(vol_shape, device)
    print("Loaded pore mask")

    component_volume = SampleComponent(
        tm.StaticTrack(2, device, vol_shape, learning_rate=0,
                       bounds=[half_size for dim in range(3) for half_size in (vol_shape[2 - dim] / 2,) * 2],
                       mask_confines=False, pore_mask=pore_mask, loss_weight=0),
        sm.VolumeArray(3, device, roi_shape, voxel_scale, learning_rate=TRAINING_PARAMS["default_lr"],
                       attenuation_range=VOLUME_ATTENUATION_RANGE, loss_weight=0))

    # Combine all projectors into one sequence and put them in the CTSimulation module that manages the trajectory
    sample_projectors = SequentialCTModule(collections.OrderedDict(
        [('static_matrix', SIRTCTSimulation(component_volume, sampling_rate=1)),
         ('nothing', CTEffects()),
         ]))
    sino_params['num_projections'] = sinogram.num_projections  # TODO this should be in CTScan somewhere
    sino_params['roi'] = roi
    ct_sample_recon = cs.CTSimulation(device, sino_params, projectors=sample_projectors)

    pore_mask.init_pore_mask([component_volume])
    component_volume.init_component(1)

    ct_sample_recon.trajectory.learning_rate = 0
    for parameter in ct_sample_recon.trajectory.parameters():
        parameter.requires_grad = False

    # define sinogram sampler
    sampler = samplers.OrderedSubsetSampler(range(sinogram.num_projections), subset_size=subset_size)

    logger = get_logger(sinogram)
    reconstructor = CTReconstruction(sinogram, ct_sample_recon, sampler, default_lr=TRAINING_PARAMS["default_lr"])
    trainer = get_trainer(reconstructor, TRAINING_PARAMS["iterations"], 1, logger, (), check_val_every_n_epoch=None)
    run_trainer(trainer, reconstructor)
    ct_sample_recon.to(device)  # PL moves back to cpu after iteration

    static_matrix = ct_sample_recon.projectors.static_matrix.shape_model  # type: sm.VolumeArray
    limits = static_matrix.limits
    reconstructed_volume = static_matrix.attenuations[0].detach().cpu().numpy()
    ft.write_tiff(recon_volume_file, reconstructed_volume, limits, np.uint16)


def main():
    """Reconstruct a static attenuation volume for each of TOTAL_FRAMES on-disk
    projection sets under the active source's scan_base."""
    os.makedirs(recon_folder, exist_ok=True)

    for i in range(TOTAL_FRAMES):
        reconstruct_frame(i, sino_params)


if __name__ == "__main__":
    main()
