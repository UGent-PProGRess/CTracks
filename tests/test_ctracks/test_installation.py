"""
Installation smoke test for the particle-tracking (ctracks) pipeline.

This does NOT check scientific accuracy - it checks that your Python/CUDA/PyTorch
installation can actually run the differentiable CT reconstruction end to end:
imports resolve, tensors move to the right device, the custom projectors and
Lightning training loop execute without error, and gradient descent actually
refines particle tracks towards their true positions.

How it works: it fabricates a small set of particles moving in straight lines
(known ground truth - no external data files needed), renders their sinogram
with the same forward projector used for real reconstructions, then reconstructs
a particle set - seeded near, but not at, the true tracks - from that sinogram
and checks that most reconstructed tracks land close to a true track (recall).

Why "near, but not at": a particle's analytic sphere patch only produces a
useful gradient very close to its own current position (see ParticleCTSimulation),
so a genuinely blind cold start (particles scattered anywhere in the volume)
depends on the same heuristic search machinery (ParticleShaker, ParticleAddition,
many more iterations, warm-starting from a previous frame, ...) that real multi-
frame reconstructions rely on - tuning that to converge reliably within a tiny
toy problem's iteration budget turned out to be exactly the kind of finicky,
scale-sensitive process that makes a poor installation check. Starting close to
the truth instead isolates the one thing this test actually needs to verify:
does the differentiable projector + optimizer loop correctly refine parameters
at all. Both are exercised by every real reconstruction; only the search part
is skipped here.

Run it directly with:
    pytest tests/test_ctracks/test_installation.py -v

A failure almost always means an installation problem (missing/mismatched
torch or CUDA build, incompatible package versions) rather than bad luck -
see the assertion message and the README's "Installation" section.
"""
import collections

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ctrex.optimization import samplers
from ctrex.optimization.reconstruction import CTReconstruction, get_trainer, get_logger, run_trainer
from ctrex.projectors import SequentialCTModule, CTNoise
from ctrex.sample_description.PoreMaskFunctions import PoreMaskPlain
from ctrex.sample_description.sample_component import SampleComponent, tm, sm
from ctrex.utils import datasets as dl, ct_setup as cs
from ctracks.particle_projectors import ParticleCTSimulation
from ctracks.update_utils.Scheduler import Scheduler

# ---------------------------------------------------------------------------
# Problem size - kept deliberately small so the test runs quickly.
# ---------------------------------------------------------------------------
SEED = 0
NUM_PARTICLES = 15
DETECTOR_HEIGHT = 64
DETECTOR_WIDTH = 64
PROJECTIONS_PER_ROTATION = 100  # angular sampling matters more than detector size here
SUBSET_SIZE = 10
ITERATIONS = 80

# Cone-beam geometry (same source-to-object/source-to-detector distances used by the
# real experimental scripts - see recon_differences_capillary_multiframe.py). Particle
# radii below are specified relative to this magnification, not in raw voxels.
SOURCE_TO_OBJECT_DISTANCE = 18.3
SOURCE_TO_DETECTOR_DISTANCE = 450
EXPECTED_MAGNIFICATION = SOURCE_TO_DETECTOR_DISTANCE / SOURCE_TO_OBJECT_DISTANCE

# Ground-truth particles are fixed spheres of this projected radius (in detector
# pixels, after magnification) and attenuation - "known particle sets" per the test's
# purpose. Reconstruction is told the true radius (radius learning rate 0 below) and
# has to recover position, velocity, and attenuation.
GT_RADIUS_FACTOR = 1.7
GT_ATTENUATION = 0.03
RECON_ATTENUATION_RANGE = (0.01, 0.1)
TRACK_LEARNING_RATE = 0.5
# Kept small relative to the 64-voxel volume: a particle drifting a large fraction of
# the frame per rotation smears across too many pixels to localise reliably at this
# scale (real datasets move a similarly small *fraction* of their much larger frames).
MAX_SYNTHETIC_SPEED = 1.5  # voxels of total displacement over the rotation, per axis
DISPLACEMENT_MAX = (6, 6, 6)  # generous vs. the max synthetic speed above
NOISE_BACKGROUND = dict(background_mean=0.259, background_std=0.012)

# Reconstruction's initial guess is the true track plus independent uniform noise of
# up to this many voxels per axis, applied separately to the start and end points -
# see the module docstring for why this isn't a blind cold start.
INIT_POSITION_NOISE_VOXELS = 3.0

# How close a reconstructed track's start/end points must be to a ground-truth
# track's (summed distance, in voxels) to count as a correct detection - mirrors
# analysis_functions.test_endpoints's `min_truth_distance` (default 1.73 there; a
# little more generous here given the shorter iteration budget).
MATCH_TOLERANCE_VOXELS = 2.0

# Minimum fraction of ground-truth particles that must be correctly recovered.
# Reconstruction involves real randomness (particle initialisation, optimizer noise),
# so this is a generous floor, not a target - a healthy installation should comfortably
# clear it.
MIN_RECALL = 0.7

# Sanity floors proving optimisation actually ran, independent of the recall check above
# (which could otherwise pass "by luck" if the initial noise itself already happened to
# land within tolerance and nothing moved from there). Both are checked against each
# particle's own starting point, well below what even a few real gradient steps produce,
# so these should never be close calls on a healthy installation.
MIN_MEAN_POSITION_CHANGE_VOXELS = 0.2  # summed movement of the start+end points
MIN_MEAN_ATTENUATION_CHANGE = 0.005  # attenuation starts randomised within RECON_ATTENUATION_RANGE


def _generate_ground_truth_tracks(rng, vol_shape, num_particles, num_projections):
    """Fabricate straight-line ground-truth tracks, one 3D position per projection.

    Particles start within a safe radius of the horizontal (y, x) center so they
    stay inside the field of view at every rotation angle, and drift linearly in
    (z, y, x) by up to `MAX_SYNTHETIC_SPEED` per axis over the rotation.
    """
    height, width, _ = vol_shape
    center = width / 2.0
    z_margin = max(10, height // 6)
    radial_max = width / 2.0 - 2 * z_margin  # stay well clear of the detector edge

    z0 = rng.uniform(z_margin, height - z_margin, num_particles)
    angle = rng.uniform(0, 2 * np.pi, num_particles)
    radius = rng.uniform(0, radial_max, num_particles)
    start = np.stack([z0, center + radius * np.cos(angle), center + radius * np.sin(angle)], axis=1)

    velocity = rng.uniform(-MAX_SYNTHETIC_SPEED, MAX_SYNTHETIC_SPEED, size=(num_particles, 3))

    t = np.linspace(0.0, 1.0, num_projections)
    tracks = start[:, None, :] + t[None, :, None] * velocity[:, None, :]  # (N, T, 3)
    return tracks.astype(np.float32)


def _render_ground_truth_sinogram(gt_tracks, rad_dist, sino_params, device):
    """Build a CTSimulation of a KnownTrack particle sample from ground-truth tracks
    and render its sinogram - mirrors recon_simulated_linear.py::generate_knowntrack_sample.
    """
    vol_shape = (sino_params['height'], sino_params['width'], sino_params['width'])
    rad_mean, rad_std, rad_range = rad_dist
    component_particles_truth = SampleComponent(
        tm.KnownTrack(sino_params['dimension'], device, vol_shape, control_points=gt_tracks),
        sm.SphereShape(sino_params['dimension'], device, attenuation_range=(GT_ATTENUATION, GT_ATTENUATION),
                       rad_mean=rad_mean, rad_std=rad_std, rad_range=rad_range))
    component_particles_truth.shape_model.init_shapes(len(gt_tracks))
    sample_projectors_gt = SequentialCTModule(collections.OrderedDict(
        [('particles', ParticleCTSimulation(component_particles_truth)),
         ('noise', CTNoise(background_type="gaussian", **NOISE_BACKGROUND))
         ]))
    ct_sample_truth = cs.CTSimulation(device, sino_params, projectors=sample_projectors_gt)

    sinogram = ct_sample_truth.empty_sinogram(dl.CTDatasetInMemory, device=device)
    sampler = samplers.OrderedSubsetSampler(range(sinogram.num_projections), subset_size=SUBSET_SIZE)
    with torch.no_grad():
        for _ in range(sampler.num_subsets):
            sampled_projections = sampler.next_projections().to(sinogram.device)
            sinogram_sample = 0 * sinogram[:, sampled_projections]
            sinogram_sample = ct_sample_truth(sinogram_sample, sampled_projections)
            sinogram[:, sampled_projections] = sinogram_sample
    return sinogram


def _greedy_match(gt_endpoints, recon_endpoints, min_truth_distance):
    """Greedily match ground-truth and reconstructed (start, end) track endpoints by
    summed distance, closest pairs first, one-to-one.

    A small self-contained stand-in for
    scripts/particle_tracking/data_analysis/analysis_functions.py::test_endpoints,
    reimplemented here so this test doesn't depend on the `scripts` package being
    importable (which needs the repo root on sys.path, unlike the installed `ctrex`/
    `ctracks` packages this test otherwise relies on).
    """
    num_gt = gt_endpoints.shape[0]
    num_recon = recon_endpoints.shape[0]
    diffs = gt_endpoints[:, None, :, :] - recon_endpoints[None, :, :, :]
    total_dists = np.linalg.norm(diffs, axis=-1).sum(axis=-1)  # (num_gt, num_recon)

    flat_order = np.argsort(total_dists, axis=None)
    gt_idx, recon_idx = np.unravel_index(flat_order, total_dists.shape)

    matched_gt = np.zeros(num_gt, dtype=bool)
    matched_recon = np.zeros(num_recon, dtype=bool)
    num_true_positives = 0
    for g, r in zip(gt_idx, recon_idx):
        if total_dists[g, r] > 2 * min_truth_distance:
            break
        if matched_gt[g] or matched_recon[r]:
            continue
        matched_gt[g] = True
        matched_recon[r] = True
        num_true_positives += 1
    return num_true_positives, num_gt, num_recon


def test_known_particle_reconstruction():
    """End-to-end smoke test: reconstruct a small set of known, moving particles - seeded
    near their true tracks - from a simulated sinogram, and check most are recovered.
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    init_rng = np.random.default_rng(SEED + 1)
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sino_params = {
        'width': DETECTOR_WIDTH,
        'height': DETECTOR_HEIGHT,
        'angles': np.linspace(0., 360., PROJECTIONS_PER_ROTATION, endpoint=True, dtype=np.float32),
        'dimension': 3,
        'centre_of_rotation': DETECTOR_WIDTH / 2,
        'sample_rate': 1,
        'skew': 0,
        'tilt': 0.0,
        'clockwise': -1,
        'sod': SOURCE_TO_OBJECT_DISTANCE,
        'sdd': SOURCE_TO_DETECTOR_DISTANCE,
        'vertical_centre': DETECTOR_HEIGHT / 2,
        'horizontal_centre': DETECTOR_WIDTH / 2,
        'binning': 1,
        'pixel_size': 0.3,
        'num_projections': PROJECTIONS_PER_ROTATION,
    }
    vol_shape = (sino_params['height'], sino_params['width'], sino_params['width'])

    rad_value = GT_RADIUS_FACTOR / EXPECTED_MAGNIFICATION
    rad_dist = (rad_value, 0.0, (rad_value, rad_value))  # (mean, std, range) - fixed known radius

    gt_tracks_np = _generate_ground_truth_tracks(rng, vol_shape, NUM_PARTICLES, PROJECTIONS_PER_ROTATION)
    gt_tracks = torch.tensor(gt_tracks_np, dtype=torch.float32, device=device)

    sinogram = _render_ground_truth_sinogram(gt_tracks, rad_dist, sino_params, device)

    # Reconstruction: seeded with the true particle count. A real reconstruction instead
    # grows/shrinks the count adaptively via ParticleAddition, but that heuristic is tuned
    # for real (much larger, noisier) datasets and can misfire badly at this toy scale
    # (repeatedly deciding to add particles it doesn't need) - deliberately left out below.
    pore_mask = PoreMaskPlain(vol_shape, device)
    component_particles_recon = SampleComponent(
        tm.LinearTrack(3, device, vol_shape, learning_rate=TRACK_LEARNING_RATE, displacement_max=DISPLACEMENT_MAX,
                       pore_mask=pore_mask, loss_weight=0),
        sm.SphereShape(sino_params['dimension'], device, attenuation_range=RECON_ATTENUATION_RANGE,
                       rad_mean=rad_dist[0], rad_std=rad_dist[1], rad_range=rad_dist[2],
                       learning_rate=(0.0, 1e-1), loss_weight=0))
    sample_projectors_recon = SequentialCTModule(collections.OrderedDict(
        [('particles', ParticleCTSimulation(component_particles_recon)),
         ('noise', CTNoise(background_type="gaussian", **NOISE_BACKGROUND))
         ]))
    ct_sample_recon = cs.CTSimulation(device, sino_params, projectors=sample_projectors_recon, extend_fov=False)

    pore_mask.init_pore_mask([component_particles_recon])
    component_particles_recon.init_component(NUM_PARTICLES)

    # Override the (otherwise uniform-random-in-volume) initial track guess with the true
    # endpoints plus independent noise - see the module docstring for why. Attenuation is
    # left at its default random init, so that channel still has real optimizing to do.
    gt_endpoints_np = gt_tracks_np[:, [0, -1], :]
    initial_guess_np = gt_endpoints_np + init_rng.uniform(
        -INIT_POSITION_NOISE_VOXELS, INIT_POSITION_NOISE_VOXELS, size=gt_endpoints_np.shape)
    with torch.no_grad():
        control_points = component_particles_recon.track_model.control_points
        control_points.data = torch.tensor(initial_guess_np, dtype=control_points.dtype, device=device)
        # Snapshot (clone!) the randomised initial attenuation so we can later confirm
        # training actually changed it - the live tensor below gets updated in place.
        initial_attenuations = component_particles_recon.shape_model.shape_params[1].detach().clone().cpu().numpy()

    ct_sample_recon.trajectory.learning_rate = 0
    for parameter in ct_sample_recon.trajectory.parameters():
        parameter.requires_grad = False

    callbacks = (
        Scheduler(ReduceLROnPlateau, mode='min', factor=0.5, patience=10),
    )
    logger = get_logger(sinogram)
    sampler = samplers.OrderedSubsetSampler(range(sinogram.num_projections), subset_size=SUBSET_SIZE)
    reconstructor = CTReconstruction(sinogram, ct_sample_recon, sampler, default_lr=TRACK_LEARNING_RATE)
    trainer = get_trainer(reconstructor, ITERATIONS, 1, logger, callbacks, check_val_every_n_epoch=None)
    run_trainer(trainer, reconstructor)
    ct_sample_recon.to(device)  # Lightning moves the model back to CPU after fitting

    with torch.no_grad():
        recon_control_points = ct_sample_recon.projectors.particles.track_model.control_points.detach().cpu().numpy()
        final_attenuations = component_particles_recon.shape_model.shape_params[1].detach().cpu().numpy()

    assert recon_control_points.shape[0] > 0, (
        "Reconstruction ended with zero particles - the pipeline ran, but something is "
        "badly wrong (check the printed training logs above for errors/NaNs)."
    )

    # Confirm optimisation actually moved parameters, rather than the recall check below
    # passing "by luck" off the initial noisy guess without any real gradient descent.
    mean_position_change = np.linalg.norm(recon_control_points - initial_guess_np, axis=-1).sum(axis=-1).mean()
    mean_attenuation_change = np.abs(final_attenuations - initial_attenuations).mean()
    print(f"Mean position change from initial guess: {mean_position_change:.3f} voxels, "
         f"mean attenuation change: {mean_attenuation_change:.4f}")

    assert mean_position_change >= MIN_MEAN_POSITION_CHANGE_VOXELS, (
        f"Particles barely moved from their (noisy) initial guess (mean summed start+end "
        f"movement {mean_position_change:.3f} voxels, need >= {MIN_MEAN_POSITION_CHANGE_VOXELS}). "
        "This suggests gradient descent isn't actually updating parameters - check for a "
        "broken backward pass, a disconnected autograd graph, or requires_grad wrongly left "
        "False (see the training log above for errors/NaNs)."
    )
    assert mean_attenuation_change >= MIN_MEAN_ATTENUATION_CHANGE, (
        f"Attenuation barely changed from its random initial guess (mean change "
        f"{mean_attenuation_change:.4f}, need >= {MIN_MEAN_ATTENUATION_CHANGE}). "
        "This suggests gradient descent isn't actually updating parameters - check for a "
        "broken backward pass or a disconnected autograd graph (see the training log above)."
    )

    recon_endpoints = recon_control_points  # LinearTrack already stores exactly 2 control points (start, end)
    num_true_positives, num_gt, num_recon = _greedy_match(gt_endpoints_np, recon_endpoints, MATCH_TOLERANCE_VOXELS)
    recall = num_true_positives / num_gt
    precision = num_true_positives / num_recon if num_recon else 0.0

    print(f"Ground-truth particles: {num_gt}, reconstructed particles: {num_recon}, "
         f"matched: {num_true_positives} (recall={recall:.1%}, precision={precision:.1%})")

    assert recall >= MIN_RECALL, (
        f"Only recovered {recall:.0%} of the {num_gt} known particles (need >= {MIN_RECALL:.0%}). "
        "This usually means something is wrong with the installation (CUDA/PyTorch build, or a "
        "package version mismatch) rather than ordinary randomness - see the README's "
        "'Installation' section, and check the training log above for warnings or NaNs."
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
