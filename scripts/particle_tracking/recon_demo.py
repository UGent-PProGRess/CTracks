"""
recon_demo.py - a beginner-friendly, single-dataset particle-tracking reconstruction.

WHAT THIS SCRIPT DOES
----------------------
You have a dynamic CT scan of a porous sample with small tracer particles flowing through
it, and you want to know where those particles were, frame by frame (one "frame" = one full
CT rotation). This script does that in three steps, for each frame in turn:

  1. Loads that frame's projections and subtracts a REFERENCE scan of the same sample with
     no particles visible - what's left over ("the difference") is mostly just the particles.
  2. Starts from a rough guess of where the particles are (either "no idea, a few hundred
     particles somewhere in the pore space" for the very first frame, or "wherever they ended
     up last frame, nudged forward by their last known velocity" for every frame after that).
  3. Iteratively adjusts each particle's position and size so that a simulated X-ray scan of
     them matches the real difference data as closely as possible, using the same maths a
     real CT reconstruction uses, just for a handful of particles instead of a whole solid
     object.

The result for each frame is a file listing every particle's position, saved under
OUTPUT_FOLDER\\ctracks_trajectories\\ - open these with the plotting/visualisation scripts
under data_analysis\\ once you have them.

HOW TO USE THIS SCRIPT
------------------------
1. Edit the "YOUR SETTINGS" section directly below to point at your own scan data and
   describe your own scan's geometry. Every setting has a comment explaining what it means
   and how to figure out the right value for your setup.
2. If you don't have a reference (particle-free) scan yet, set GENERATE_REFERENCE = True and
   run the script once - it will build one for you and then stop. Set it back to False
   afterwards.
3. Run the script (e.g. `python recon_demo.py`, or press Run in your IDE/PyCharm).

Everything below "YOUR SETTINGS" is internal machinery that you shouldn't need to touch to
reconstruct your own data - it's commented throughout in case you want to understand or
adapt it, but it's organised into functions specifically so you can ignore it and still use
the script.

This script is a simplified, single-dataset version of recon_differences_porous_multiframe.py
(which reconstructs several real lab datasets at once, selected from a dictionary, and has
several extra knobs relevant to that specific experimental setup) - use that one as a
reference for more advanced options once you're comfortable with this one.
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
from ctracks.update_utils import Scheduler as sc, ParticleShaker as ps, ParticleAddition as pa
from ctracks.postprocessing import postProcessResults as ppr
from ctrex.utils.filetools import save_projections_from_sample


# =============================================================================
# YOUR SETTINGS
# Edit everything in this section to match your own scan. Defaults below are a real,
# working example (a porous-glass sample at a 180 nl/min flow rate) so you can run the
# script as-is first, to check everything is installed and working, before pointing it at
# your own data.
# =============================================================================

# ---- Where is your data? ---------------------------------------------------

# Folder containing the raw, numbered projection tiffs of your DYNAMIC scan (the scan where
# particles are actually moving/flowing). This folder must also contain the flat-field
# ("io000001.tif") and dark-field ("di000001.tif") calibration images used to correct for
# the scanner's own brightness pattern - if your calibration files have different names,
# edit the `io_file`/`di_file` lines inside main() below.
SCAN_FOLDER = r"J:\CTracksSubmissionData\Raw data\Experimental\Porous glass\180nlmin"

# The filename pattern for the numbered projection tiffs in SCAN_FOLDER, with "%s" where the
# projection number goes - e.g. "180nlmin_%s.tif" matches 180nlmin_000001.tif, ...000002.tif,
# and so on.
SCAN_FILENAME_FORMAT = "180nlmin_%s.tif"

# Folder containing a REFERENCE scan of the exact same sample with no moving particles
# visible in it (e.g. a scan taken before particles were introduced). If you don't have one,
# see GENERATE_REFERENCE below.
REFERENCE_FOLDER = r"J:\CTracksSubmissionData\Processed results\Experimental\Porous glass\180nlmin\background_projections"

# A segmented tiff volume marking which voxels are pore space (where a particle CAN
# physically be) versus solid grain material (where it can't). This keeps tracked particles
# physically plausible, and is normally produced by segmenting a static, high-quality scan
# of the empty sample in whatever software you usually use for that.
MASK_FILE = r"J:\CTracksSubmissionData\Processed results\Experimental\Porous glass\segmentation.tif"

# Folder to write results into (reconstructed trajectories, and optionally debug images -
# see SAVE_DEBUG_PROJECTIONS below). It will be created automatically if it doesn't exist.
OUTPUT_FOLDER = r"J:\CTracksSubmissionData\Processed results\Experimental\Porous glass\180nlmin"

# If you don't yet have a reference scan for REFERENCE_FOLDER, set this to True and run the
# script once. It builds one automatically from SCAN_FOLDER, by keeping only the brightest
# pixel seen at each detector position across every repeat of the same viewing angle (0
# degrees appears once per rotation, and so on) - since a moving particle is only bright at
# one detector position at a time, this "brightest pixel wins" trick suppresses particles and
# leaves behind a clean, particle-free reference. The script will save it and then stop -
# set this back to False afterwards to actually reconstruct particle trajectories.
GENERATE_REFERENCE = False


# ---- Describe your scan's geometry -----------------------------------------
# These come from your CT scanner and its acquisition settings for this particular scan -
# check your scanner's acquisition software/logs for the exact values. Getting these right
# matters a lot: they define exactly how a 3D position maps onto the detector.

DETECTOR_HEIGHT = 948              # full detector height, in pixels
DETECTOR_WIDTH = 560                # full detector width, in pixels
PROJECTIONS_PER_ROTATION = 850      # number of projections making up one full 360-degree rotation
TOTAL_ROTATIONS = 60                # how many rotations (frames) of SCAN_FOLDER to reconstruct

SOURCE_TO_OBJECT_DISTANCE = 18.3    # source-to-object distance ("SOD"), in your scan's length unit (e.g. cm)
SOURCE_TO_DETECTOR_DISTANCE = 450   # source-to-detector distance ("SDD"), same unit as above
PIXEL_SIZE = 0.3                    # detector pixel size, same unit as above

CENTRE_OF_ROTATION = 279.12         # detector column (in pixels) that the rotation axis passes through
VERTICAL_CENTRE = 474.503           # detector row considered the vertical centre (usually DETECTOR_HEIGHT / 2)
HORIZONTAL_CENTRE = 280             # detector column considered the horizontal centre (usually DETECTOR_WIDTH / 2)
DETECTOR_TILT = -0.02               # detector tilt, in degrees, if your rig has a small known tilt (else just use 0)

# A region of interest (ROI) crops the detector to a smaller area before reconstruction, to
# save memory/time - e.g. if you know your sample only occupies the middle of the detector.
# Use slice(0, DETECTOR_HEIGHT) / slice(0, DETECTOR_WIDTH) if you don't want to crop anything.
ROI_HEIGHT = slice(224, 724)        # which detector rows to use
ROI_WIDTH = slice(80, 480)          # which detector columns to use

# Some projections may be contaminated by something crossing the field of view that isn't
# part of your sample (e.g. the tubing that delivers fluid/particles) - list their indices
# here (within one rotation, i.e. between 0 and PROJECTIONS_PER_ROTATION - 1) to exclude them
# from the reconstruction. Use an empty list, [], if this doesn't apply to your setup.
EXCLUDED_PROJECTIONS = list(range(0, 160)) + list(range(445, 535))


# ---- Describe your particles ------------------------------------------------

# Roughly how many particles you expect to be visible in the reconstructed volume at once.
# This is just a starting point - the reconstruction will automatically add more particles
# later on if this guess turns out to be too low (see setup_callbacks below).
EXPECTED_PARTICLE_COUNT = 600

# The largest distance (in voxels, along x, y, z) a particle is allowed to move between one
# projection and the next. Set this comfortably above your fastest expected particle speed -
# too low will clip genuine fast motion, too high makes tracking less stable/precise.
MAX_PARTICLE_DISPLACEMENT = (7.5, 7.5, 15)

# Expected particle attenuation range (roughly, how strongly a particle absorbs X-rays
# relative to the surrounding fluid) as a (minimum, maximum) pair. The default range below is
# a good starting point for most tracer-particle materials; only change it if you know your
# particles are unusually faint or dense.
PARTICLE_ATTENUATION_RANGE = (0.01, 0.1)


# Whether to also save the intermediate difference/reconstructed projection images (as tiffs,
# alongside the trajectories) - useful the first few times you run this on a new dataset, so
# you can visually check the difference sinogram looks sensible and the reconstruction is
# actually matching it. Turn this off once you trust your settings, to save disk space/time.
SAVE_DEBUG_PROJECTIONS = True


# =============================================================================
# Everything below this point is internal machinery - you shouldn't need to change it to
# reconstruct your own data (that's what the settings above are for). It's organised into
# functions and commented throughout, in case you want to understand or adapt it.
# =============================================================================

# ---- Advanced/rarely-changed settings --------------------------------------
# These almost never differ between scans on the same rig, so they're kept separate from
# "YOUR SETTINGS" above to avoid clutter - but they're genuine settings, not magic numbers,
# so they're still named and commented here rather than buried inside a function.
ROTATION_DIRECTION = -1             # +1 or -1, whichever matches your scanner's rotation direction
DETECTOR_SKEW = 0                   # detector skew, in degrees (0 unless you know otherwise)
DETECTOR_BINNING = 1                # detector pixel binning factor used during acquisition

TRACK_LEARNING_RATE = 0.5           # how fast particle positions are allowed to update per training step
TRAINING_ITERATIONS = 100           # how many passes to make over the data when reconstructing each frame
SUBSET_SIZE = 20                    # how many projections are grouped into one "ordered subset" update
CENTRE_OF_ROTATION_LEARNING_RATE = 0.1  # allows small automatic refinement of CENTRE_OF_ROTATION per frame
NOISE_ESTIMATE_WIDTH = 20           # how many pixels of border (on each side) are used to estimate background noise
NOISE_ESTIMATE_MAX_PROJECTIONS = 500  # caps how many projections are used for that noise estimate, for speed


def build_sino_params():
    """Assemble the geometry dict ("sino_params") that the reconstruction engine expects,
    directly from the YOUR SETTINGS / advanced settings above. You shouldn't need to edit
    this function - edit the settings themselves instead."""
    return {
        'width': DETECTOR_WIDTH,
        'height': DETECTOR_HEIGHT,
        'angles': np.linspace(0., 360., PROJECTIONS_PER_ROTATION, endpoint=False, dtype=np.float32),
        'last_angle': 360.,
        'dimension': 3,
        'centre_of_rotation': CENTRE_OF_ROTATION,
        'sample_rate': 1,
        'skew': DETECTOR_SKEW,
        'tilt': DETECTOR_TILT,
        'clockwise': ROTATION_DIRECTION,
        'sod': SOURCE_TO_OBJECT_DISTANCE,
        'sdd': SOURCE_TO_DETECTOR_DISTANCE,
        'vertical_centre': VERTICAL_CENTRE,
        'horizontal_centre': HORIZONTAL_CENTRE,
        'binning': DETECTOR_BINNING,
        'pixel_size': PIXEL_SIZE,
        'num_projections': PROJECTIONS_PER_ROTATION,
    }


def build_particle_model(device, sino_params, vol_shape, roi):
    """Build the particle model that will be fitted to each frame's difference sinogram: a
    physical description of "up to a few hundred small spheres, each following its own
    straight-line path within a frame, confined to the segmented pore space".

    Returns (component_particles, sample_projectors, pore_mask).
    """
    # Particle radius (in voxels) is derived from the scan's magnification - particles
    # typically span 0.8x-2.5x their true size once magnified onto the detector, and this
    # spread is converted to a std. deviation via the "full width at a tenth of maximum"
    # (FWTM) rule of thumb (dividing by 4.29).
    magnification = SOURCE_TO_DETECTOR_DISTANCE / SOURCE_TO_OBJECT_DISTANCE
    rad_range = (0.8 / magnification, 2.5 / magnification)
    rad_mean = 1 / magnification
    rad_std = (rad_range[1] - rad_range[0]) / 4.29

    print("Loading pore mask")
    mask = tifffile.imread(MASK_FILE) > 0.5
    mask = torch.from_numpy(mask).to(device).to(torch.bool)
    voi = (roi[0], roi[1], roi[1])
    pore_mask = PoreMaskFunctions(mask, voi=voi)
    print("Loaded pore mask")

    component_particles = SampleComponent(
        tm.LinearTrack(3, device, vol_shape, learning_rate=TRACK_LEARNING_RATE,
                       displacement_max=MAX_PARTICLE_DISPLACEMENT, pore_mask=pore_mask, loss_weight=0),
        sm.SphereShape(sino_params['dimension'], device, attenuation_range=PARTICLE_ATTENUATION_RANGE,
                       rad_mean=rad_mean, rad_std=rad_std, rad_range=rad_range,
                       learning_rate=(1e-1, 1e-1), loss_weight=0))

    sample_projectors = SequentialCTModule(collections.OrderedDict(
        [('particles', ParticleCTSimulation(component_particles)),
         ('noise', CTNoise(background_type="gaussian"))  # noise level is estimated automatically per frame below
         ]))
    pore_mask.init_pore_mask([component_particles])
    return component_particles, sample_projectors, pore_mask


def setup_callbacks():
    """Build the small set of automated helpers that run during each frame's reconstruction:
      - proi (ParticleROITools): periodically removes particles whose track has entirely left
        the visible ROI (never projects inside it, in any view) and respawns an equal number
        of new particles near existing ones - recapturing particles that would otherwise
        silently drift out of view for good.
      - shaker (ParticleShaker): every so often, adds a small random jitter to particle
        positions/sizes, helping the optimisation escape local optima.
      - scheduler (Scheduler): lowers the learning rate once reconstruction stops improving,
        so late-stage refinement takes smaller, more careful steps.
      - add_particles (ParticleAddition): periodically checks whether the current particles
        explain the data well, and adds more if a lot of signal is still unaccounted for.
      - postprocess (ParticlePostProcess): removes low-confidence particles sitting right at
        the edge of the reconstructed volume, which tend to be unreliable.
    You shouldn't need to change these unless you're fine-tuning reconstruction quality.
    """
    proi = ParticleROITools()
    shaker = ps.ParticleShaker(interval=20, decay=[0.95, 0.5], width=[[3, 3, 3], [0, 0]])
    scheduler = sc.Scheduler(ReduceLROnPlateau, mode='min', factor=0.5, patience=20)
    add_particles = pa.ParticleAddition(interval=30, intensity_threshold=0.8)
    postprocess = ppr.ParticlePostProcess(edge_range=1, att_removal_fraction=0.1)
    return (proi, scheduler, shaker, add_particles, postprocess)


# System / technical setup - not user-facing, leave alone unless you know what you're doing.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # https://discuss.pytorch.org/t/how-to-fix-cuda-error-device-side-assert-triggered-error/137553/11
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('highest')
print(torch.version.__version__)
print("device =", device, type(device))


def main():
    """Reconstruct particle trajectories for every frame of SCAN_FOLDER, one CT rotation at
    a time, saving each frame's result before moving on to the next. Particle positions are
    carried over between frames (extrapolated forward by their last known velocity), so later
    frames start from the previous frame's solution instead of from scratch."""
    io_file = os.path.join(SCAN_FOLDER, "io000001.tif")
    di_file = os.path.join(SCAN_FOLDER, "di000001.tif")

    if GENERATE_REFERENCE:
        # Build a particle-free reference scan from SCAN_FOLDER and stop - see the comment
        # on GENERATE_REFERENCE above for how this works.
        ft.create_reference_scan(SCAN_FOLDER, REFERENCE_FOLDER, SCAN_FILENAME_FORMAT, io_file, di_file,
                                 PROJECTIONS_PER_ROTATION, TOTAL_ROTATIONS)
        print("Reference scan generated - set GENERATE_REFERENCE = False and run again to reconstruct.")
        sys.exit(0)

    sino_params = build_sino_params()
    roi = (ROI_HEIGHT, ROI_WIDTH)
    vol_shape = (DETECTOR_HEIGHT, DETECTOR_WIDTH, DETECTOR_WIDTH)
    mem_projections = PROJECTIONS_PER_ROTATION  # how many projections to keep in (GPU) memory at once

    # The reference scan is loaded once and re-used (subtracted) for every frame.
    sinogram_ref = dl.CTDatasetOnDisk((REFERENCE_FOLDER, 'proj_%s.tif'), slice(0, None, 1), torch.device('cpu'),
                                      mem_projections=None,  # all
                                      flat_field=None, dark_field=None, roi=roi,
                                      fixed_memory=mem_projections - SUBSET_SIZE,
                                      verbose=False)

    component_particles, sample_projectors, pore_mask = build_particle_model(device, sino_params, vol_shape, roi)

    ct_sample_recon = None  # built once, on the first frame - see below
    for frame_index in range(TOTAL_ROTATIONS):
        print(f"--- Tracking frame {frame_index + 1}/{TOTAL_ROTATIONS} ---")

        # Load this frame's dynamic-scan projections and subtract the reference, leaving
        # (mostly) just the particles.
        sinogram_dyn = dl.CTDatasetOnDisk((SCAN_FOLDER, SCAN_FILENAME_FORMAT),
                                          slice(frame_index * PROJECTIONS_PER_ROTATION,
                                                (frame_index + 1) * PROJECTIONS_PER_ROTATION, 1), device,
                                          mem_projections=mem_projections,
                                          flat_field=ft.load_tiff(io_file)[0], dark_field=ft.load_tiff(di_file)[0],
                                          roi=roi, fixed_memory=mem_projections - SUBSET_SIZE, verbose=False)
        sinogram_frame = dl.CTDatasetDifference(sinogram_ref, sinogram_dyn)

        if SAVE_DEBUG_PROJECTIONS:
            print("Saving difference projections")
            ft.save_projections(sinogram_frame, folder=OUTPUT_FOLDER,
                                subfolder=os.path.join("saved_projections", "difference", f"scan_{frame_index}"))
        else:
            # Not saving to disk - the underlying dataset still needs every projection to be
            # loaded (and cached) once before reconstruction can sample from it efficiently.
            for pi in range(sinogram_frame.num_projections):
                projection = sinogram_frame[:, pi]

        if ct_sample_recon is None:
            # First frame only: build the reconstruction model from the sinogram's actual
            # geometry, and seed it with EXPECTED_PARTICLE_COUNT randomly-placed particles.
            #
            # sino_params['angles'] is built with endpoint=False (see build_sino_params), so
            # the true last angle is just short of 360 degrees - correct last_angle to match,
            # exactly like the reconstruction scripts this one is based on do.
            updated_sino_params = dict(sino_params)
            updated_sino_params['last_angle'] = float(sino_params['angles'][-1])
            mem_projections_recon = min(mem_projections, sinogram_frame.num_projections)
            projection_times = torch.arange(sinogram_frame.num_projections, dtype=torch.long, device=device)
            # No projection averaging is used here, so excluded-projection indices need no
            # remapping - they refer to the same positions in sinogram_frame as in the raw scan.
            frame_excluded_projections = EXCLUDED_PROJECTIONS

            ct_sample_scan = cs.CTScan(sinogram_frame, updated_sino_params)
            scan_sino_params = ct_sample_scan.trajectory.sino_params
            ct_sample_recon = cs.CTSimulation(device, scan_sino_params, projectors=sample_projectors, extend_fov=False)
            for parameter in ct_sample_recon.trajectory.parameters():
                parameter.requires_grad = False  # scan geometry itself isn't being optimised (see CENTRE_OF_ROTATION below)

            component_particles.init_component(EXPECTED_PARTICLE_COUNT)
        else:
            # Every later frame: carry particles forward from where the last frame left them,
            # extrapolated by their most recent velocity, instead of starting from scratch.
            recon_tracks = ct_sample_recon.projectors.particles.track_model.control_points.data
            recon_velocity = recon_tracks[:, -1, :] - recon_tracks[:, 0, :]
            recon_tracks += recon_velocity.unsqueeze(1)

        callbacks = setup_callbacks()

        # Estimate background noise level directly from this frame's own data (the very
        # edges of the detector, away from the sample, should contain only noise).
        ct_sample_recon.projectors.noise.estimate_noise_from_sides(
            sinogram_frame, width=NOISE_ESTIMATE_WIDTH,
            num_projections=min(NOISE_ESTIMATE_MAX_PROJECTIONS, sinogram_frame.num_projections))

        if SAVE_DEBUG_PROJECTIONS:
            save_projections_from_sample(ct_sample_recon, sinogram_frame,
                                         (OUTPUT_FOLDER, os.path.join("saved_projections", "init_difference_raw", f"scan_{frame_index}")),
                                         scan_sino_params, projection_times, mem_projections, SUBSET_SIZE)

        # Allow a small amount of automatic centre-of-rotation refinement each frame, in case
        # CENTRE_OF_ROTATION above is slightly off - everything else about the scan geometry
        # stays fixed (see the requires_grad = False loop above).
        ct_sample_recon.trajectory.learning_rate = {'centre_of_rotation': CENTRE_OF_ROTATION_LEARNING_RATE}
        ct_sample_recon.trajectory.centre_of_rotation.requires_grad = True

        # This is the actual reconstruction step: iteratively adjust particle positions/sizes
        # so the simulated sinogram matches sinogram_frame as closely as possible.
        logger = get_logger(sinogram_frame)
        sampler = samplers.OrderedSubsetSampler(range(sinogram_frame.num_projections), subset_size=SUBSET_SIZE,
                                                disabled_projections=frame_excluded_projections)
        reconstructor = CTReconstruction(sinogram_frame, ct_sample_recon, sampler, default_lr=TRACK_LEARNING_RATE)
        trainer = get_trainer(reconstructor, TRAINING_ITERATIONS, 1, logger, callbacks, check_val_every_n_epoch=None)
        run_trainer(trainer, reconstructor)
        ct_sample_recon.to(device)  # Lightning moves the model back to cpu after each frame

        print("Centre of rotation:", ct_sample_recon.trajectory.centre_of_rotation)
        print("Saving result trajectories" + (" and projections" if SAVE_DEBUG_PROJECTIONS else ""))

        ft.save_trajectories(recon_component=ct_sample_recon.projectors.particles.sample_component,
                             folder=os.path.join(OUTPUT_FOLDER, "ctracks_trajectories"), filename=f"scan_{frame_index}")

        if SAVE_DEBUG_PROJECTIONS:
            save_projections_from_sample(ct_sample_recon, sinogram_frame,
                                         (OUTPUT_FOLDER, os.path.join("saved_projections", "recon_difference", f"scan_{frame_index}")),
                                         scan_sino_params, projection_times, mem_projections_recon, SUBSET_SIZE, normalise=True)


if __name__ == "__main__":
    main()
