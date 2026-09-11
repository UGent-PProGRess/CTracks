"""
Reconstruct the differences in a dynamic scan with respect to a reference scan.
These differences are supposed to be the particles.

Multi-frame reconstruction for the capillary datasets: each CT rotation ("frame") of
the dynamic scan is reconstructed independently, with particle positions from the
previous frame used to initialise the next (extrapolated by their velocity).
"""

import collections
import os
import sys

import numpy as np
import tifffile
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ctracks.update_utils.ParticleROITools import ParticleROITools
from ctrex.optimization import samplers
from ctrex.utils import datasets as dl, filetools as ft, ct_setup as cs
from ctrex.sample_description.sample_component import SampleComponent, tm, sm
from ctrex.sample_description.PoreMaskFunctions import PoreMaskFunctions
from ctrex.projectors import *
from ctrex.optimization.reconstruction import CTReconstruction, get_trainer, get_logger, run_trainer
from ctracks.particle_projectors import ParticleCTSimulation
from ctracks.postprocessing import postProcessResults as ppr
from ctracks.update_utils import Scheduler as sc, ParticleShaker as ps, ParticleAddition as pa
from ctrex.utils.filetools import save_projections_from_sample

# ---------------------------------------------------------------------------
# Run-mode flags
# ---------------------------------------------------------------------------
save_projections = True
save_results = True

# ---------------------------------------------------------------------------
# Data root - change this if CTracksSubmissionData is moved or mounted under a
# different drive/path (e.g. moving to a new PC).
# ---------------------------------------------------------------------------
DATA_ROOT = "J:\\"

# ---------------------------------------------------------------------------
# Capillary scan geometry (shared by every dataset reconstructed by this script)
# ---------------------------------------------------------------------------
SCAN_GEOMETRY = dict(
    height=948,
    proj_per_rot=850,
    width=360,
    sod=18.3,
    sdd=450,
    vertical_centre=474.911,
    horizontal_centre=180,
    tilt=0,
    mask_file=os.path.join(DATA_ROOT, r'CTracksSubmissionData\Processed results\Experimental\Capillary\segmentation.tif'),
)
# ROI selection kept as the pre-existing manual list-index toggle: swap the trailing
# index to pick a different predefined ROI option.
SCAN_GEOMETRY["roi_height"] = [
    slice(int((SCAN_GEOMETRY["height"] - 500) // 2), int((SCAN_GEOMETRY["height"] + 500) // 2)),
    slice(0, 800),
][1]
SCAN_GEOMETRY["roi_width"] = [
    slice(100, 250),
    slice(0, SCAN_GEOMETRY["width"]),
][0]

# ---------------------------------------------------------------------------
# Acquisition parameters (feed directly into sino_params below)
# ---------------------------------------------------------------------------
ACQUISITION_PARAMS = dict(
    dimension=3,
    last_angle=360.,
    sample_rate=1,
    skew=0,
    clockwise=-1,
    binning=1,
    pixel_size=0.3,
)

# ---------------------------------------------------------------------------
# Per-dataset (flow-rate) scan configuration - select the active one below
# ---------------------------------------------------------------------------
DATASETS = {
    "60nlmin": dict(
        recon_name="recon_60nlmin",
        scan_fmt='60nlmin_%s.tif',
        scan_folder_full=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Capillary\60nlmin"),
        velocity_max=(5, 5, 10),
        flow_line_projections=list(range(60, 120)) + list(range(470, 550)),
        centre_of_rotation=179.62,
        total_scan_rotations=40,
        scan_folder_base=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Capillary\60nlmin"),
        projection_averaging=1,
        scan_folder_ref=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Capillary\60nlmin\background_projections"),
    ),
    "180nlmin": dict(
        recon_name="recon_180nlmin",
        scan_fmt='180nlmin_%s.tif',
        scan_folder_full=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Capillary\180nlmin"),
        velocity_max=(5, 5, 24),
        flow_line_projections=list(range(45, 130)) + list(range(430, 580)),
        centre_of_rotation=179.12,
        total_scan_rotations=40,
        scan_folder_base=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Capillary\180nlmin"),
        projection_averaging=1,
        scan_folder_ref=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Capillary\180nlmin\background_projections"),
    ),
    "1ulmin": dict(
        recon_name="recon_1ulmin",
        scan_fmt='1ulmin_%s.tif',
        scan_folder_full=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Capillary\1ulmin"),
        velocity_max=(5, 5, 120),
        flow_line_projections=list(range(45, 130)) + list(range(430, 580)),
        centre_of_rotation=178.44,
        total_scan_rotations=20,
        scan_folder_base=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Capillary\1ulmin"),
        projection_averaging=1,
        scan_folder_ref=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Capillary\1ulmin\background_projections"),
    ),
}
ACTIVE_DATASET = "180nlmin"

# ---------------------------------------------------------------------------
# Particle initialisation parameters
# ---------------------------------------------------------------------------
PARTICLE_INIT = dict(
    initial_particle_count=600,
    attenuation_range=(0.01, 0.1),
    shape_learning_rate=(1e-1, 1e-1),
    rad_mean=1,
    rad_min_factor=0.8,
    rad_max_factor=2.5,
    rad_std_divisor=4.29,  # FWTM
)
TRACK_LEARNING_RATE = 0.5

# ---------------------------------------------------------------------------
# Callback hyperparameters (see setup_callbacks)
# ---------------------------------------------------------------------------
CALLBACK_PARAMS = dict(
    intensity_threshold=0.8, #particle addition
    shaker_interval=20,
    shaker_decay=[0.95, 0.5], #track,shape
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
    subset_base=20,
    centre_of_rotation_learning_rate=0.1,
    noise_estimate_width=20,
    noise_estimate_max_projections=500,
)


cfg = DATASETS[ACTIVE_DATASET]
recon_name = cfg["recon_name"]
scan_fmt = cfg["scan_fmt"]
scan_folder_full = cfg["scan_folder_full"]
velocity_max = cfg["velocity_max"]
flow_line_projections = cfg["flow_line_projections"]
centre_of_rotation = cfg["centre_of_rotation"]
total_scan_rotations = cfg["total_scan_rotations"]
scan_folder_base = cfg["scan_folder_base"]
projection_averaging = cfg["projection_averaging"]
scan_folder_ref = cfg["scan_folder_ref"]

# ---------------------------------------------------------------------------
# Sinogram parameters passed to the CT reconstruction (built from SCAN_GEOMETRY /
# ACQUISITION_PARAMS above and the active dataset's centre_of_rotation)
# ---------------------------------------------------------------------------
sino_params = {
    'width': SCAN_GEOMETRY["width"],
    'height': SCAN_GEOMETRY["height"],
    'angles': np.linspace(0., 360., SCAN_GEOMETRY["proj_per_rot"], endpoint=False, dtype=np.float32),  # TODO check if endpoint included with 850 vs 851
    'last_angle': ACQUISITION_PARAMS["last_angle"],
    'dimension': ACQUISITION_PARAMS["dimension"],
    'centre_of_rotation': centre_of_rotation,
    'sample_rate': ACQUISITION_PARAMS["sample_rate"],
    'skew': ACQUISITION_PARAMS["skew"],
    'tilt': SCAN_GEOMETRY["tilt"],
    'clockwise': ACQUISITION_PARAMS["clockwise"],
    'sod': SCAN_GEOMETRY["sod"],
    'sdd': SCAN_GEOMETRY["sdd"],
    'vertical_centre': SCAN_GEOMETRY["vertical_centre"],  # detector height / 2
    'horizontal_centre': SCAN_GEOMETRY["horizontal_centre"],
    'binning': ACQUISITION_PARAMS["binning"],
    'pixel_size': ACQUISITION_PARAMS["pixel_size"],
    'num_projections': SCAN_GEOMETRY["proj_per_rot"]
}

def setup_callbacks(averages=1):
    """Build the PyTorch-Lightning callbacks used during per-frame reconstruction
    (ROI-based respawn, LR scheduling, particle jitter, adaptive particle addition,
    and post-hoc filtering).

    Args:
        averages: number of raw projections averaged into each reconstructed
            projection. The particle-addition intensity threshold is halved when
            averaging is used, since averaging reduces the visible particle intensity.
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


# ---------------------------------------------------------------------------
# System / technical setup (not user-facing - leave alone unless you know what
# you're doing)
# ---------------------------------------------------------------------------
# use this line for debugging
# https://discuss.pytorch.org/t/how-to-fix-cuda-error-device-side-assert-triggered-error/137553/11
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.set_float32_matmul_precision('high')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.version.__version__)
print("device =", device, type(device))


def main():
    """Run the multi-frame particle reconstruction for the active capillary dataset.

    Builds the reference/dynamic sinogram sources, pore mask and particle model once,
    then reconstructs each CT rotation ("frame") of the dynamic scan in turn, saving
    per-frame trajectories and projection images. Particle positions are carried over
    between frames (extrapolated by velocity) so that later frames start from the
    previous frame's solution.
    """
    io_file = os.path.join(scan_folder_full, "io000001.tif")
    di_file = os.path.join(scan_folder_full, "di000001.tif")

    scan_folder_dyn = scan_folder_full


    # Define the acquired CT dataset
    subset_size = TRAINING_PARAMS["subset_base"] // projection_averaging
    mem_projections = SCAN_GEOMETRY["proj_per_rot"]  # during the reconstruction, keep this many projections in (GPU) memory

    roi = (SCAN_GEOMETRY["roi_height"], SCAN_GEOMETRY["roi_width"])
    sinogram_ref = dl.CTDatasetOnDisk((scan_folder_ref, 'proj_%s.tif'), slice(0, None, 1), torch.device('cpu'),
                                      mem_projections=None,  # all
                                      flat_field=None, dark_field=None, roi=roi,
                                      fixed_memory=mem_projections - subset_size,
                                      verbose=False)  # roi gets truncated to full width

    # set up particle sizes
    expected_magnification = sino_params['sdd'] / sino_params['sod']
    rad_range = (PARTICLE_INIT["rad_min_factor"] / expected_magnification, PARTICLE_INIT["rad_max_factor"] / expected_magnification)
    rad_mean = PARTICLE_INIT['rad_mean'] / expected_magnification
    rad_std = (rad_range[1] - rad_range[0]) / PARTICLE_INIT["rad_std_divisor"]  # FWTM

    # Set up pore mask
    vol_shape = (SCAN_GEOMETRY["height"], SCAN_GEOMETRY["width"], SCAN_GEOMETRY["width"])
    print("Loading pore mask")
    mask = tifffile.imread(SCAN_GEOMETRY["mask_file"]) > 0.5
    mask = torch.from_numpy(mask).to(device).to(torch.bool)
    voi = (roi[0], roi[1], roi[1])
    pore_mask = PoreMaskFunctions(mask, voi=voi)
    print("Loaded pore mask")

    # set up reconstructed ct sample
    learning_rate = TRACK_LEARNING_RATE
    component_particles = SampleComponent(
        tm.LinearTrack(3, device, vol_shape, learning_rate=learning_rate, displacement_max=velocity_max,
                       pore_mask=pore_mask, loss_weight=0),
        sm.SphereShape(sino_params['dimension'], device, attenuation_range=PARTICLE_INIT["attenuation_range"],
                       rad_mean=rad_mean, rad_std=rad_std, rad_range=rad_range,
                       learning_rate=PARTICLE_INIT["shape_learning_rate"], loss_weight=0))

    sample_projectors = SequentialCTModule(collections.OrderedDict(
        [('particles', ParticleCTSimulation(component_particles)),
         ('noise', CTNoise(background_type="gaussian"))  # noise values estimated later
         ]))
    pore_mask.init_pore_mask([component_particles])

    recon_rot_range = range(0, total_scan_rotations)
    for i in recon_rot_range:
        print(f"Tracking frame {i}")
        sinogram_dyn = dl.CTDatasetOnDisk((scan_folder_dyn, scan_fmt),
                                          slice(i * SCAN_GEOMETRY["proj_per_rot"], (i + 1) * SCAN_GEOMETRY["proj_per_rot"], 1), device,
                                          mem_projections=mem_projections,
                                          flat_field=ft.load_tiff(io_file)[0], dark_field=ft.load_tiff(di_file)[0], roi=roi,
                                          fixed_memory=mem_projections - subset_size,
                                          verbose=False)

        sinogram_diff = dl.CTDatasetDifference(sinogram_ref, sinogram_dyn)
        sinogram_average = dl.CTDatasetAverage(sinogram_diff, projection_averaging)

        if save_projections:
            print("Saving normalised difference projections")
            ft.save_projections(sinogram_average, folder=scan_folder_base, subfolder=rf'saved_projections\difference\scan_{i}')
            print("Saving non-normalized difference projections")
            ft.save_projections(sinogram_average, folder=scan_folder_base, subfolder=rf'saved_projections\difference_raw\scan_{i}', normalise=False)
        if not save_projections:
            for pi in range(sinogram_average.num_projections):  # seems like loading one at a time is better - not sure why
                projection = sinogram_average[:, pi]

        if i == recon_rot_range[0]:

            # update sino params, angles, etc. if averaging
            updated_sino_params = sinogram_average.update_sino_params_average(sino_params)
            mem_projections_recon = min(mem_projections, sinogram_average.num_projections)
            projection_times = torch.arange(sinogram_average.num_projections, dtype=torch.long, device=device)
            frame_flow_line_projections = sinogram_average.convert_disabled_projections(flow_line_projections)

            ct_sample_scan = cs.CTScan(sinogram_average, updated_sino_params)  # computes trajectory with initialized sino_params
            scan_sino_params = ct_sample_scan.trajectory.sino_params
            ct_sample_recon = cs.CTSimulation(device, scan_sino_params, projectors=sample_projectors, extend_fov=False)

            for parameter in ct_sample_recon.trajectory.parameters():
                parameter.requires_grad = False

            component_particles.init_component(PARTICLE_INIT["initial_particle_count"])
        else:
            recon_tracks = ct_sample_recon.projectors.particles.track_model.control_points.data
            recon_vel = recon_tracks[:, -1, :] - recon_tracks[:, 0, :]
            recon_tracks += recon_vel.unsqueeze(1)
        callbacks = setup_callbacks(averages=projection_averaging)

        ct_sample_recon.projectors.noise.estimate_noise_from_sides(
            sinogram_average, width=TRAINING_PARAMS["noise_estimate_width"],
            num_projections=min(TRAINING_PARAMS["noise_estimate_max_projections"], sinogram_average.num_projections))

        if save_projections:
            save_projections_from_sample(ct_sample_recon, sinogram_average,
                                         (scan_folder_base, rf'saved_projections\init_difference_raw\scan_{i}'), scan_sino_params,
                                         projection_times, mem_projections_recon, subset_size)

        ct_sample_recon.trajectory.learning_rate = {'centre_of_rotation': TRAINING_PARAMS["centre_of_rotation_learning_rate"]}
        ct_sample_recon.trajectory.centre_of_rotation.requires_grad = True

        logger = get_logger(sinogram_average)
        sampler = samplers.OrderedSubsetSampler(range(sinogram_average.num_projections), subset_size=subset_size,
                                                disabled_projections=frame_flow_line_projections)
        reconstructor = CTReconstruction(sinogram_average, ct_sample_recon, sampler, default_lr=learning_rate)
        trainer = get_trainer(reconstructor, TRAINING_PARAMS["iterations"], 1, logger, callbacks, check_val_every_n_epoch=None)
        run_trainer(trainer, reconstructor)
        ct_sample_recon.to(device)  # PL moves back to cpu after iteration

        print("Center of rotation:", ct_sample_recon.trajectory.centre_of_rotation)
        print("Saving result trajectories and projections")

        if save_results:
            ft.save_trajectories(recon_component=ct_sample_recon.projectors.particles.sample_component,
                                 folder=os.path.join(scan_folder_base, "ctracks_trajectories"), filename=f"scan_{i}")

        if save_projections:
            print("Saving normalised reconstructed difference projections")
            save_projections_from_sample(ct_sample_recon, sinogram_average,
                                         (scan_folder_base, rf'saved_projections\recon_difference\scan_{i}'), scan_sino_params,
                                         projection_times, mem_projections_recon, subset_size, normalise=True)
            print("Saving non-normalized reconstructed difference projections")
            save_projections_from_sample(ct_sample_recon, sinogram_average,
                                         (scan_folder_base, rf'saved_projections\recon_difference_raw\scan_{i}'), scan_sino_params,
                                         projection_times, mem_projections_recon, subset_size, normalise=False)


if __name__ == "__main__":
    main()
