"""
Reconstruct particle tracks against a self-generated ground-truth sinogram.

Unlike the recon_differences_* scripts (which reconstruct against a difference
sinogram loaded from real dynamic-scan tiffs), this script builds its own ground-truth
sinogram from simulated particle tracks (loaded from a CSV, or a cached .npy of
previously-loaded tracks) via a KnownTrack model, then reconstructs a LinearTrack model
against it frame by frame, saving a ground-truth-vs-reconstruction comparison per frame.
"""

import collections
import copy
import os
import numpy as np
import torch
from tifffile import tifffile

from ctracks.update_utils.ParticleROITools import ParticleROITools
from ctrex.optimization import samplers
from ctrex.utils import datasets as dl, filetools as ft, ct_setup as cs
from ctrex.sample_description.sample_component import SampleComponent, tm, sm
from ctrex.sample_description.PoreMaskFunctions import PoreMaskFunctions
from ctrex.projectors import *
from ctrex.optimization.reconstruction import CTReconstruction, get_trainer, get_logger, run_trainer
from ctracks.particle_projectors import ParticleCTSimulation
from ctracks.update_utils import (Scheduler as sc, ParticleShaker as ps, ParticleAddition as pa)
from ctracks.postprocessing import postProcessResults as ppr
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------------------------------------------------------
# Data root - change this if CTracksSubmissionData is moved or mounted under a
# different drive/path (e.g. moving to a new PC).
# ---------------------------------------------------------------------------
DATA_ROOT = "J:\\"

recon_name = "simulated_piecewise_linear"

# ---------------------------------------------------------------------------
# Simulated ground-truth source (OpenFOAM particle tracks + pore mask) - select the
# active one below.
# ---------------------------------------------------------------------------
SIMULATIONS = {
    "porevisco_27Xfaster": dict(
        mask_file=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Simulation\fields\mask.tif"),
        track_file=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Simulation\tracks_for_projection.csv"),
        scan_folder=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Simulated"),
        run_prefix="porevisco-27Xfaster",
    ),
}
ACTIVE_SIMULATION = "porevisco_27Xfaster"
sim_cfg = SIMULATIONS[ACTIVE_SIMULATION]
mask_file = sim_cfg["mask_file"]
track_file = sim_cfg["track_file"]
scan_folder = sim_cfg["scan_folder"]
run_prefix = sim_cfg["run_prefix"]

# cache of the sampled ground-truth tracks so re-runs don't need to re-load/re-sample the CSV
past_gt_file = os.path.join(scan_folder, "ground_truth_tracks.npy")

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

subset_size = 20
mem_projections = num_projections

# ---------------------------------------------------------------------------
# Particle initialisation parameters
# ---------------------------------------------------------------------------
PARTICLE_INIT_GT = dict(
    attenuation_range=(0.03, 0.03),
    rad_factor=1.7,  # fixed radius (rad_std=0), as a multiple of the expected magnification
    rad_std=0,  # fixed radius (alternative: derive from rad_range via FWTM divisor 4.29)
)
PARTICLE_INIT_RECON = dict(
    initial_particle_fraction=0.25,  # fraction of ground-truth particle count to seed the reconstruction with
    displacement_max=(90, 90, 90),
    attenuation_range=(0.01, 0.1),
    shape_learning_rate=(0, 1e-1),
)
TRACK_LEARNING_RATE = 0.5

# maximum number of frames (CT rotations) to reconstruct, even if more ground-truth data is available
MAX_RECON_ROTATIONS = 100

# ---------------------------------------------------------------------------
# Callback hyperparameters (see setup_callbacks)
# ---------------------------------------------------------------------------
CALLBACK_PARAMS = dict(
    intensity_threshold=0.8,
    shaker_interval=20,
    shaker_decay=[0.95, 0.5],
    shaker_width=[[3, 3, 3], [0, 0]],
    scheduler_factor=0.5,
    scheduler_patience=20,
    add_particles_interval=30,
    postprocess_edge_range=1,
    postprocess_att_removal_fraction=0.1,
)

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
TRAINING_PARAMS = dict(
    iterations=100,
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

NOISE_BACKGROUND = dict(background_mean=0.259, background_std=0.012)
noise_projector_gt = CTNoise(background_type="gaussian", **NOISE_BACKGROUND)
noise_projector_recon = CTNoise(background_type="gaussian", **NOISE_BACKGROUND)

# ---------------------------------------------------------------------------
# System / technical setup (not user-facing - leave alone unless you know what
# you're doing)
# ---------------------------------------------------------------------------
np.random.seed(4)  # For reproducibility
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')  # or medium/highest?
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.version.__version__)
print("device =", device, type(device))


@torch.no_grad()
def generate_knowntrack_sample(tracks, rad_dist, sino_params):
    """Build a CTSimulation of a KnownTrack particle sample from ground-truth tracks
    and render its ground-truth sinogram.

    Args:
        tracks: ground-truth particle control points for this frame.
        rad_dist: (rad_mean, rad_std, rad_range) particle radius distribution.
        sino_params: sinogram/geometry parameters for the simulation.

    Returns:
        (ct_sample_truth, sinogram): the ground-truth CTSimulation and its rendered sinogram.
    """
    vol_shape = (sino_params['height'], sino_params['width'], sino_params['width'])
    device = tracks.device
    rad_mean, rad_std, rad_range = rad_dist
    component_particles_truth = SampleComponent(
        tm.KnownTrack(sino_params['dimension'], device, vol_shape, control_points=tracks),
        sm.SphereShape(sino_params['dimension'], device, attenuation_range=PARTICLE_INIT_GT["attenuation_range"],
                       rad_mean=rad_mean, rad_std=rad_std, rad_range=rad_range))
    component_particles_truth.shape_model.init_shapes(len(tracks))
    sample_projectors_gt = SequentialCTModule(collections.OrderedDict(
        [('particles', ParticleCTSimulation(component_particles_truth)),
         ('noise', copy.deepcopy(noise_projector_gt))
         ]))
    ct_sample_truth = cs.CTSimulation(device, sino_params, projectors=sample_projectors_gt)

    # Generate the ground truth sinogram directly
    sinogram = ct_sample_truth.empty_sinogram(dl.CTDatasetInMemory, device=device)
    sampler = samplers.OrderedSubsetSampler(range(sinogram.num_projections), subset_size=subset_size)
    for si in range(sampler.num_subsets):
        sampled_projections = sampler.next_projections().to(sinogram.device)
        sinogram_sample = 0 * sinogram[:, sampled_projections]
        sinogram_sample = ct_sample_truth(sinogram_sample, sampled_projections)
        sinogram[:, sampled_projections] = sinogram_sample
    return ct_sample_truth, sinogram


def setup_callbacks(ct_sample, averages=1, prev_callbacks=None, eval_detailed=False):
    """Build the PyTorch-Lightning callbacks used during per-frame reconstruction
    (ROI-based respawn, LR scheduling, particle jitter, adaptive particle addition,
    and post-hoc filtering).

    Args:
        ct_sample: the ground-truth CTSimulation for the current frame (currently unused,
            kept for interface symmetry with callers).
        averages: number of raw projections averaged into each reconstructed
            projection. The particle-addition intensity threshold is halved when
            averaging is used, since averaging reduces the visible particle intensity.
        prev_callbacks: callbacks from the previous frame (currently unused).
        eval_detailed: whether to run detailed evaluation (currently unused).
    """
    proi = ParticleROITools()
    intensity_threshold = CALLBACK_PARAMS["intensity_threshold"]
    if averages > 1:
        intensity_threshold /= 2  # TODO look at
    shaker = ps.ParticleShaker(interval=CALLBACK_PARAMS["shaker_interval"],
                               decay=CALLBACK_PARAMS["shaker_decay"],
                               width=CALLBACK_PARAMS["shaker_width"])
    scheduler = sc.Scheduler(ReduceLROnPlateau, mode='min', factor=CALLBACK_PARAMS["scheduler_factor"],
                             patience=CALLBACK_PARAMS["scheduler_patience"])
    add_particles = pa.ParticleAddition(interval=CALLBACK_PARAMS["add_particles_interval"],
                                        intensity_threshold=intensity_threshold)
    postprocess = ppr.ParticlePostProcess(edge_range=CALLBACK_PARAMS["postprocess_edge_range"],
                                          att_removal_fraction=CALLBACK_PARAMS["postprocess_att_removal_fraction"])
    return (proi, scheduler, shaker, add_particles, postprocess)


def main():
    """Generate a ground-truth sinogram from simulated particle tracks for each CT
    rotation, then reconstruct a particle model against it and save trajectories
    comparing the reconstruction to ground truth.
    """
    os.makedirs(scan_folder, exist_ok=True)

    # set up expected distributions
    expected_magnification = sino_params['sdd'] / sino_params['sod']
    gt_rad_range = (PARTICLE_INIT_GT["rad_factor"] / expected_magnification, PARTICLE_INIT_GT["rad_factor"] / expected_magnification)
    recon_rad_range = gt_rad_range

    rad_mean = (gt_rad_range[0] + gt_rad_range[1]) / 2
    rad_std = PARTICLE_INIT_GT["rad_std"]

    gt_rad_dist = (rad_mean, rad_std, gt_rad_range)

    # Set up pore mask
    zero_bounds = (3, 3, 3)
    mask = tifffile.imread(mask_file) > 0.5
    mask = torch.from_numpy(mask).to(device).to(torch.bool)
    pore_mask = PoreMaskFunctions(mask, zero_bounds=zero_bounds)
    print("Loaded pore mask")
    vol_shape = pore_mask.shape

    # set up ground truth
    print("Loading particle ground truth")
    if os.path.isfile(past_gt_file):
        all_gt_tracks = torch.from_numpy(np.load(past_gt_file)).to(device)
    else:
        particle_crop_ranges = ((0, pore_mask.shape[2]), (0, pore_mask.shape[1]), (20, pore_mask.shape[0] - 20))
        all_gt_tracks, _ = ft.load_simulated_tracks(track_file, device, 1000, valid_range=particle_crop_ranges)
        np.save(past_gt_file, all_gt_tracks.cpu().numpy())
    print("Loaded particle ground truth")
    total_rotations = all_gt_tracks.shape[1] // SCAN_GEOMETRY["proj_per_rot"]

    # set up reconstructed ct sample
    component_particles_recon = SampleComponent(
        tm.LinearTrack(3, device, vol_shape, learning_rate=TRACK_LEARNING_RATE, displacement_max=PARTICLE_INIT_RECON["displacement_max"],
                       pore_mask=pore_mask, loss_weight=0),
        sm.SphereShape(sino_params['dimension'], device, attenuation_range=PARTICLE_INIT_RECON["attenuation_range"],
                       rad_mean=rad_mean, rad_std=rad_std, rad_range=recon_rad_range,
                       learning_rate=PARTICLE_INIT_RECON["shape_learning_rate"], loss_weight=0))

    sample_projectors_recon = SequentialCTModule(collections.OrderedDict(
        [('particles', ParticleCTSimulation(component_particles_recon)),
         ('nothing', CTEffects()),
         ('noise', noise_projector_recon)
         ]))
    ct_sample_recon = cs.CTSimulation(device, sino_params, projectors=sample_projectors_recon, extend_fov=False)

    pore_mask.init_pore_mask([component_particles_recon])
    component_particles_recon.init_component(int(len(all_gt_tracks) * PARTICLE_INIT_RECON["initial_particle_fraction"]))

    ct_sample_recon.trajectory.learning_rate = 0
    for parameter in ct_sample_recon.trajectory.parameters():
        parameter.requires_grad = False

    callbacks = None
    for i in range(total_rotations):
        print("Generating ground truth sinogram")
        gt_tracks = all_gt_tracks[:, i * SCAN_GEOMETRY["proj_per_rot"]:(i + 1) * SCAN_GEOMETRY["proj_per_rot"], :]
        ct_sample_truth, sinogram = generate_knowntrack_sample(gt_tracks, gt_rad_dist, sino_params)
        ft.save_projections(sinogram, scan_folder, os.path.join("saved_projections", f"proj_rot{i}"))
        print("Generated ground truth sinogram")
        if i >= MAX_RECON_ROTATIONS:
            break
        if i == 0:
            callbacks = setup_callbacks(ct_sample_truth)
        else:
            callbacks = setup_callbacks(ct_sample_truth, prev_callbacks=callbacks, eval_detailed=True)
            with torch.no_grad():
                recon_tracks = ct_sample_recon.projectors.particles.track_model.control_points.data
                recon_vel = recon_tracks[:, -1, :] - recon_tracks[:, 0, :]
                recon_tracks += recon_vel.unsqueeze(1)

        logger = get_logger(sinogram)
        projection_indices = range(sinogram.num_projections)
        sampler = samplers.OrderedSubsetSampler(projection_indices, subset_size=subset_size)
        reconstructor = CTReconstruction(sinogram, ct_sample_recon, sampler, gt_sample=ct_sample_truth, default_lr=TRACK_LEARNING_RATE)
        trainer = get_trainer(reconstructor, TRAINING_PARAMS["iterations"], 1, logger, callbacks, check_val_every_n_epoch=None)
        run_trainer(trainer, reconstructor)
        ct_sample_recon.to(device)  # PL moves back to cpu after iteration

        ft.save_trajectories(ct_sample_truth.projectors.particles.sample_component,
                             ct_sample_recon.projectors.particles.sample_component,
                             os.path.join(scan_folder, "ctracks_trajectories"), f"params_rot{i}")
        ft.save_projections_from_sample(ct_sample_recon, sinogram,
                                     (scan_folder, os.path.join("saved_projections", f"proj_recon_rot{i}")), sino_params,
                                     sinogram.projection_times, mem_projections, subset_size, normalise=False)


if __name__ == "__main__":
    main()
