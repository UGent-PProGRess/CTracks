"""
Realistic capillary-flow integration test for the particle-tracking pipeline.

Unlike test_installation.py (a fast, narrowly-scoped smoke test), this exercises the
pipeline the way scripts/particle_tracking/recon_differences_capillary_multiframe.py
actually uses it: a genuine pore-mask-confined capillary geometry, a cold (blind)
particle init, and every heuristic callback (ParticleROITools, Scheduler,
ParticleShaker, ParticleAddition, ParticlePostProcess) at their real, non-toy-shrunken
hyperparameters - not test_installation.py's stripped-down, warm-started subset. It is
correspondingly slower and, being closer to the real thing, more sensitive to
reconstruction-quality randomness; treat it as a deeper confidence check, not a routine
install check (use test_installation.py for that - it's marked `slow` for this reason,
see pyproject.toml's pytest markers).

Still no real data: a synthetic capillary tube (a cylinder pore mask) is generated,
seeded with particles undergoing ideal Poiseuille flow (v(r) = v_max * (1 - (r/R)^2)
along the tube axis), and reconstructed from scratch from the resulting simulated
sinogram - the same kind of validation
scripts/particle_tracking/data_analysis/analyse_capillary.py does against real
experimental data (see its plot_velocity_radial_distribution), just against a known
ground truth instead of a nominal syringe flow rate. The reconstructed particles'
radial velocity profile is fit to a Poiseuille curve and compared to the (known, exact)
theoretical one; a plot is saved to tests/test_ctracks/output/ for visual inspection.

Ground truth is used ONLY to validate the result (recall, and the theoretical profile to
compare against) - never to pick which reconstructed particles feed the profile fit. A
real reconstruction's particle count doesn't equal the true count (ParticleAddition can
overshoot substantially, and removal is disabled - see ParticleAddition.py), and the
extras are disproportionately spurious/under-converged; but in a real experiment there is
no ground truth to filter them against. So this test doesn't use one either: it rejects
local velocity outliers (`_reject_velocity_outliers`, a self-contained cousin of
link_and_filter_functions.py's `uod_filter`) using only each reconstructed particle's own
position and velocity - the same principle the real analysis pipeline relies on for
exactly this problem.

Coordinate convention (verified against the source, not assumed - see
track_models.py::StaticTrack.clamp_params, PoreMaskFunctions.random_nonzero, and
PoreMaskFunctions.find_edge_detections): particle positions/tracks are (x, y, z), with
x and y bounded by `width` and z (last) bounded by `height` - the rotation axis. Mask
ARRAYS are indexed the other way round, (z, y, x), matching
`vol_shape = (height, width, width)`. This capillary's tube axis is z; the pore mask is
a cylinder in the (x, y) plane, constant along z.

Run it directly with:
    pytest tests/test_ctracks/test_capillary_integration.py -v -m slow
"""
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless - this test only ever saves a plot, never shows one
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ctrex.optimization import samplers
from ctrex.optimization.reconstruction import CTReconstruction, get_trainer, get_logger, run_trainer
from ctrex.projectors import SequentialCTModule, CTNoise
from ctrex.sample_description.PoreMaskFunctions import PoreMaskFunctions
from ctrex.sample_description.sample_component import SampleComponent, tm, sm
from ctrex.utils import datasets as dl, ct_setup as cs
from ctracks.particle_projectors import ParticleCTSimulation
from ctracks.postprocessing.postProcessResults import ParticlePostProcess
from ctracks.update_utils.ParticleAddition import ParticleAddition
from ctracks.update_utils.ParticleROITools import ParticleROITools
from ctracks.update_utils.ParticleShaker import ParticleShaker
from ctracks.update_utils.Scheduler import Scheduler

OUTPUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# Problem size. Larger and slower than test_installation.py by design - this is a
# thorough integration check, not a quick per-commit smoke test.
# ---------------------------------------------------------------------------
SEED = 0
DETECTOR_HEIGHT = 200  # z / rotation axis / capillary length
DETECTOR_WIDTH = 100   # x, y / capillary cross-section
PROJECTIONS_PER_ROTATION = 200
SUBSET_SIZE = 20
ITERATIONS = 150

NUM_GT_PARTICLES = 150
INITIAL_RECON_PARTICLES = 60  # a deliberately-low guess; ParticleAddition should grow it

# Cone-beam geometry - same source-to-object/source-to-detector distances used by the
# real experimental scripts (see recon_differences_capillary_multiframe.py).
SOURCE_TO_OBJECT_DISTANCE = 18.3
SOURCE_TO_DETECTOR_DISTANCE = 450
EXPECTED_MAGNIFICATION = SOURCE_TO_DETECTOR_DISTANCE / SOURCE_TO_OBJECT_DISTANCE

# Detector ROI: cropped in x only (matching the real scripts' convention), comfortably
# containing the capillary - this is what gives ParticleROITools an actual restriction
# to check against, instead of the no-op it is in test_installation.py.
ROI_X = (10, 90)

# --- Capillary geometry -----------------------------------------------------------
CAPILLARY_RADIUS = 35.0  # voxels, in the (x, y) plane, centered on the volume
CAPILLARY_CENTER = DETECTOR_WIDTH / 2.0
CAPILLARY_Z_MARGIN = 15  # keep particles this far from the top/bottom of the volume

# --- Flow: ideal Poiseuille profile along z, v(r) = POISEUILLE_V_MAX * (1 - (r/R)^2) ---
POISEUILLE_V_MAX = 6.0  # voxels of z-displacement over the rotation, at the tube center

# --- Particle appearance (same scale as the real scripts' capillary/porous setups) -----
GT_RADIUS_FACTOR = 1.7
GT_ATTENUATION = 0.03
RECON_ATTENUATION_RANGE = (0.01, 0.1)
TRACK_LEARNING_RATE = 0.5
DISPLACEMENT_MAX = (4, 4, 15)  # (x, y, z) - generous vs. POISEUILLE_V_MAX and ~0 transverse flow
NOISE_BACKGROUND = dict(background_mean=0.259, background_std=0.012)

# --- Callback hyperparameters (mirrors the real scripts' setup_callbacks almost verbatim) --
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

# --- Ground-truth-free outlier rejection (see module docstring for why this exists
# instead of matching against ground truth) - a single-frame, position/velocity-only
# cousin of link_and_filter_functions.py's uod_filter: reject particles whose z-velocity
# deviates too far from their spatial neighbors' median, relative to the neighbors'
# own local spread (so it adapts to how noisy each region actually is, rather than using
# one fixed cutoff everywhere).
VELOCITY_OUTLIER_K_NEIGHBORS = 8
VELOCITY_OUTLIER_THRESHOLD = 3.0  # multiples of local median-absolute-deviation
VELOCITY_OUTLIER_EPS = 0.5  # regularizer preventing division by ~0 in very uniform regions

# --- Radial velocity profile fit ---------------------------------------------------
NUM_RADIAL_BINS = 10
TRIM_RADIUS = CAPILLARY_RADIUS * 0.9  # drop the noisiest near-wall bin, as the real analysis does

# Relative error allowed between the fitted and theoretical Poiseuille coefficients. This
# is a real (not warm-started) reconstruction, so it's noisier than test_installation.py.
MAX_RELATIVE_COEFFICIENT_ERROR = 0.20
MIN_RECALL = 0.5  # coarse sanity floor, not the primary check (the profile fit is)
MATCH_TOLERANCE_VOXELS = 3.0

# Sanity floors proving optimisation actually did the work, independent of the recall/
# profile checks above. INITIAL_RECALL_CEILING rules out the reconstruction "succeeding"
# merely because enough randomly-placed particles were seeded/added that some inevitably
# land near a true one by chance, with no real refinement - measured on the blind random
# init, BEFORE any training step, so it should be low regardless of problem difficulty
# (the tube volume is far larger than the match tolerance's capture radius around each
# true particle). MIN_RECALL_IMPROVEMENT then requires training to have actually closed
# most of the gap from that low starting point up to the final recall.
INITIAL_RECALL_CEILING = 0.15
MIN_RECALL_IMPROVEMENT = 0.3


def _poiseuille_velocity(r, v_max, radius):
    return v_max * (1.0 - (r / radius) ** 2)


def _build_capillary_mask(height, width, radius, center):
    """A cylinder pore mask: True inside `radius` of `center` in the (x, y) plane, for
    every z. Array-indexed (z, y, x) to match `vol_shape = (height, width, width)`.
    """
    yy, xx = np.meshgrid(np.arange(width), np.arange(width), indexing='ij')
    disk = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    return np.broadcast_to(disk, (height, width, width)).copy()


def _generate_ground_truth_tracks(rng, num_particles, height, num_projections):
    """Fabricate ideal-Poiseuille-flow ground-truth tracks, one (x, y, z) position per
    projection. Particles are placed uniformly across the tube's radius (not
    area-weighted) so the profile is well sampled all the way to the wall, and drift at
    constant z-velocity given by `_poiseuille_velocity` - true laminar flow has no
    transverse (x, y) velocity, so there is none here either.

    Returns:
        (tracks, radii): tracks has shape (N, num_projections, 3); radii is each
            particle's (fixed) true radial distance from the tube's centerline.
    """
    r = rng.uniform(0, CAPILLARY_RADIUS * 0.95, num_particles)  # stay just inside the wall
    theta = rng.uniform(0, 2 * np.pi, num_particles)
    x0 = CAPILLARY_CENTER + r * np.cos(theta)
    y0 = CAPILLARY_CENTER + r * np.sin(theta)
    z0 = rng.uniform(CAPILLARY_Z_MARGIN, height - CAPILLARY_Z_MARGIN, num_particles)

    v_z = _poiseuille_velocity(r, POISEUILLE_V_MAX, CAPILLARY_RADIUS)

    t = np.linspace(0.0, 1.0, num_projections)
    start = np.stack([x0, y0, z0], axis=1)
    velocity = np.stack([np.zeros_like(v_z), np.zeros_like(v_z), v_z], axis=1)
    tracks = start[:, None, :] + t[None, :, None] * velocity[:, None, :]  # (N, T, 3), xyz
    return tracks.astype(np.float32), r


def _render_ground_truth_sinogram(gt_tracks, rad_dist, sino_params, device):
    """Build a CTSimulation of a KnownTrack particle sample from ground-truth tracks and
    render its sinogram - mirrors recon_simulated_linear.py::generate_knowntrack_sample.
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


def _setup_callbacks():
    """Mirrors recon_differences_capillary_multiframe.py::setup_callbacks - all five
    heuristic callbacks a real reconstruction runs, at their real (not toy-shrunken)
    hyperparameters.
    """
    return (
        ParticleROITools(),
        Scheduler(ReduceLROnPlateau, mode='min', factor=CALLBACK_PARAMS["scheduler_factor"],
                 patience=CALLBACK_PARAMS["scheduler_patience"]),
        ParticleShaker(interval=CALLBACK_PARAMS["shaker_interval"], decay=CALLBACK_PARAMS["shaker_decay"],
                      width=CALLBACK_PARAMS["shaker_width"]),
        ParticleAddition(interval=CALLBACK_PARAMS["add_particles_interval"],
                         intensity_threshold=CALLBACK_PARAMS["intensity_threshold"]),
        ParticlePostProcess(edge_range=CALLBACK_PARAMS["postprocess_edge_range"],
                            att_removal_fraction=CALLBACK_PARAMS["postprocess_att_removal_fraction"]),
    )


def _greedy_match(gt_endpoints, recon_endpoints, min_truth_distance):
    """Greedily match ground-truth and reconstructed (start, end) track endpoints by
    summed distance, closest pairs first, one-to-one - see test_installation.py for the
    full explanation of why this is reimplemented rather than imported from
    analysis_functions.py.

    Used here ONLY to report recall/precision as a diagnostic - never to decide which
    reconstructed particles feed the profile fit (see the module docstring for why: a
    real experiment has no ground truth to filter against, so this test doesn't rely on
    one for that either).
    """
    num_gt = gt_endpoints.shape[0]
    num_recon = recon_endpoints.shape[0]
    diffs = gt_endpoints[:, None, :, :] - recon_endpoints[None, :, :, :]
    total_dists = np.linalg.norm(diffs, axis=-1).sum(axis=-1)

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


def _reject_velocity_outliers(positions_xy, v_z, k_neighbors, threshold, eps):
    """Reject particles whose z-velocity deviates too far from their k-nearest (in x, y)
    spatial neighbors' median velocity, relative to those neighbors' own local spread
    (median absolute deviation) - a single-frame, ground-truth-free stand-in for
    link_and_filter_functions.py's uod_filter. Physically motivated rather than
    arbitrary: real Poiseuille flow varies smoothly with radius, so genuine particles
    track their neighbors' velocity; spurious/under-converged ones (see the module
    docstring) generally don't.

    Returns a boolean keep-mask, all-True if there aren't enough particles to form a
    neighborhood.
    """
    n = len(v_z)
    if n < k_neighbors + 1:
        return np.ones(n, dtype=bool)

    tree = cKDTree(positions_xy)
    _, indices = tree.query(positions_xy, k=k_neighbors + 1)
    neighbor_v = v_z[indices[:, 1:]]  # (n, k) - excludes the particle itself

    median_v = np.median(neighbor_v, axis=1)
    residual = np.abs(v_z - median_v)
    neighbor_residual = np.abs(neighbor_v - median_v[:, None])
    local_spread = np.median(neighbor_residual, axis=1)

    return residual / (local_spread + eps) < threshold


def _fit_and_plot_radial_profile(r, v_z, save_path):
    """Bin reconstructed particles by radius, fit a Poiseuille curve (v(r) = C*(R^2-r^2),
    R fixed at the known true radius) to the binned means, plot both against the exact
    theoretical curve, and return the fitted coefficient C.

    Mirrors the binning/fitting approach in
    scripts/particle_tracking/data_analysis/analyse_capillary.py::plot_velocity_radial_distribution,
    simplified since the true radius and true coefficient are known exactly here (no
    syringe-flow-rate unit conversion needed).
    """
    bins = np.linspace(0, CAPILLARY_RADIUS, NUM_RADIAL_BINS + 1)
    bin_indices = np.digitize(r, bins)
    bin_centers, bin_means, bin_counts = [], [], []
    for i in range(1, len(bins)):
        in_bin = bin_indices == i
        if np.any(in_bin):
            bin_centers.append((bins[i - 1] + bins[i]) / 2)
            bin_means.append(v_z[in_bin].mean())
            bin_counts.append(in_bin.sum())
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_counts = np.array(bin_counts)

    def poiseuille_fixed_radius(radius_arr, coefficient):
        return coefficient * (CAPILLARY_RADIUS ** 2 - radius_arr ** 2)

    fit_mask = bin_centers <= TRIM_RADIUS
    popt, _ = curve_fit(poiseuille_fixed_radius, bin_centers[fit_mask], bin_means[fit_mask],
                        p0=[POISEUILLE_V_MAX / CAPILLARY_RADIUS ** 2],
                        sigma=1.0 / np.sqrt(bin_counts[fit_mask]))
    fitted_coefficient = popt[0]

    r_curve = np.linspace(0, CAPILLARY_RADIUS, 200)
    theoretical_coefficient = POISEUILLE_V_MAX / CAPILLARY_RADIUS ** 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(r, v_z, s=10, alpha=0.25, color='gray', label='Reconstructed particles (outlier-filtered)')
    ax.scatter(bin_centers, bin_means, s=60, color='blue', edgecolors='black', zorder=5,
              label='Binned mean')
    ax.plot(r_curve, poiseuille_fixed_radius(r_curve, fitted_coefficient), color='blue',
           linewidth=2, label='Fitted Poiseuille profile')
    ax.plot(r_curve, poiseuille_fixed_radius(r_curve, theoretical_coefficient), color='black',
           linestyle='--', linewidth=2, label='Theoretical Poiseuille profile')
    ax.set_xlabel('Radial distance from tube center (voxels)')
    ax.set_ylabel('z-velocity (voxels/rotation)')
    ax.set_title('Reconstructed vs. theoretical Poiseuille radial velocity profile')
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    return fitted_coefficient, theoretical_coefficient


@pytest.mark.slow
def test_capillary_poiseuille_reconstruction():
    """Full-pipeline integration test: reconstruct a synthetic Poiseuille capillary flow
    from scratch (all callbacks, genuine pore-mask confinement) and check the recovered
    radial velocity profile is close to the known theoretical one.
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
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
        'roi': (slice(0, DETECTOR_HEIGHT), slice(*ROI_X)),
    }
    vol_shape = (sino_params['height'], sino_params['width'], sino_params['width'])

    rad_value = GT_RADIUS_FACTOR / EXPECTED_MAGNIFICATION
    rad_dist = (rad_value, 0.0, (rad_value, rad_value))

    gt_tracks_np, _ = _generate_ground_truth_tracks(rng, NUM_GT_PARTICLES, DETECTOR_HEIGHT, PROJECTIONS_PER_ROTATION)
    gt_tracks = torch.tensor(gt_tracks_np, dtype=torch.float32, device=device)

    sinogram = _render_ground_truth_sinogram(gt_tracks, rad_dist, sino_params, device)

    capillary_mask = torch.from_numpy(_build_capillary_mask(
        DETECTOR_HEIGHT, DETECTOR_WIDTH, CAPILLARY_RADIUS, CAPILLARY_CENTER)).to(device)
    pore_mask = PoreMaskFunctions(capillary_mask, zero_bounds=(2, 2, 2))

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
    # Deliberately blind: a genuine cold start within the tube, relying on the callbacks
    # below (ParticleShaker to escape a bad initial guess, ParticleAddition to grow
    # towards the true count) - unlike test_installation.py, which warm-starts.
    component_particles_recon.init_component(INITIAL_RECON_PARTICLES)

    # Snapshot the blind random init, before any training step, to later confirm training
    # is what closes the gap to the true tracks - not just having enough randomly-placed
    # particles that some inevitably land near a true one by chance (see
    # INITIAL_RECALL_CEILING's docstring above).
    with torch.no_grad():
        initial_control_points = component_particles_recon.track_model.control_points.detach().clone().cpu().numpy()

    ct_sample_recon.trajectory.learning_rate = 0
    for parameter in ct_sample_recon.trajectory.parameters():
        parameter.requires_grad = False

    callbacks = _setup_callbacks()
    logger = get_logger(sinogram)
    sampler = samplers.OrderedSubsetSampler(range(sinogram.num_projections), subset_size=SUBSET_SIZE)
    reconstructor = CTReconstruction(sinogram, ct_sample_recon, sampler, default_lr=TRACK_LEARNING_RATE)
    trainer = get_trainer(reconstructor, ITERATIONS, 1, logger, callbacks, check_val_every_n_epoch=None)
    run_trainer(trainer, reconstructor)
    ct_sample_recon.to(device)

    with torch.no_grad():
        recon_control_points = ct_sample_recon.projectors.particles.track_model.control_points.detach().cpu().numpy()

    assert recon_control_points.shape[0] > 0, (
        "Reconstruction ended with zero particles - the pipeline ran, but something is "
        "badly wrong (check the printed training logs above for errors/NaNs)."
    )

    gt_endpoints = gt_tracks_np[:, [0, -1], :]

    num_true_positives_initial, num_gt_initial, num_recon_initial = _greedy_match(
        gt_endpoints, initial_control_points, MATCH_TOLERANCE_VOXELS)
    initial_recall = num_true_positives_initial / num_gt_initial
    print(f"Blind initial random guess (before any training): {num_recon_initial} particles, "
         f"recall={initial_recall:.1%}")
    assert initial_recall <= INITIAL_RECALL_CEILING, (
        f"The blind random initial guess already recalled {initial_recall:.0%} of the "
        f"{num_gt_initial} known particles (expected <= {INITIAL_RECALL_CEILING:.0%}) - before any "
        "training step ran. That means this test's particle counts/tolerances make it "
        "possible to 'pass' by chance rather than by real optimisation; increase "
        "MATCH_TOLERANCE_VOXELS's strictness or reduce INITIAL_RECON_PARTICLES rather than "
        "trusting the recall/profile checks below."
    )

    num_true_positives, num_gt, num_recon = _greedy_match(
        gt_endpoints, recon_control_points, MATCH_TOLERANCE_VOXELS)
    recall = num_true_positives / num_gt
    precision = num_true_positives / num_recon if num_recon else 0.0
    print(f"Ground-truth particles: {num_gt}, reconstructed particles: {num_recon}, "
         f"matched: {num_true_positives} (recall={recall:.1%}, precision={precision:.1%})")
    assert recall - initial_recall >= MIN_RECALL_IMPROVEMENT, (
        f"Recall only improved from {initial_recall:.0%} (blind initial guess) to {recall:.0%} "
        f"over the whole training run (need an improvement of >= {MIN_RECALL_IMPROVEMENT:.0%}). "
        "This suggests optimisation isn't actually refining particle tracks - check for a "
        "broken backward pass, a disconnected autograd graph, or requires_grad wrongly left "
        "False (see the training log above for errors/NaNs)."
    )
    assert recall >= MIN_RECALL, (
        f"Only recovered {recall:.0%} of the {num_gt} known particles (need >= {MIN_RECALL:.0%}), "
        "which is too few for the radial velocity profile below to be meaningful. Check the "
        "training log above for errors/NaNs - or this reconstruction's hyperparameters "
        "(iterations, callback intervals) may simply need retuning for this problem size."
    )

    # Reject local velocity outliers before fitting - ground-truth-free (see module
    # docstring): a real reconstruction's particle count doesn't equal the true count
    # (ParticleAddition can overshoot substantially, and removal is disabled - see
    # ParticleAddition.py), and the extras are disproportionately under-converged/
    # spurious. This uses only each particle's own position and velocity, exactly as a
    # real analysis (with no ground truth available) would have to.
    recon_x_all = recon_control_points[:, :, 0].mean(axis=1)
    recon_y_all = recon_control_points[:, :, 1].mean(axis=1)
    recon_v_z_all = recon_control_points[:, 1, 2] - recon_control_points[:, 0, 2]

    keep_mask = _reject_velocity_outliers(
        np.stack([recon_x_all, recon_y_all], axis=1), recon_v_z_all,
        VELOCITY_OUTLIER_K_NEIGHBORS, VELOCITY_OUTLIER_THRESHOLD, VELOCITY_OUTLIER_EPS)
    print(f"Velocity-outlier filter kept {keep_mask.sum()}/{len(keep_mask)} reconstructed particles")

    recon_x, recon_y, recon_v_z = recon_x_all[keep_mask], recon_y_all[keep_mask], recon_v_z_all[keep_mask]
    recon_r = np.sqrt((recon_x - CAPILLARY_CENTER) ** 2 + (recon_y - CAPILLARY_CENTER) ** 2)

    save_path = OUTPUT_DIR / "capillary_poiseuille_profile.png"
    fitted_coefficient, theoretical_coefficient = _fit_and_plot_radial_profile(recon_r, recon_v_z, save_path)
    relative_error = abs(fitted_coefficient - theoretical_coefficient) / theoretical_coefficient

    print(f"Fitted Poiseuille coefficient: {fitted_coefficient:.6f}, theoretical: "
         f"{theoretical_coefficient:.6f} (relative error {relative_error:.1%})")
    print(f"Radial velocity profile plot saved to {save_path}")

    assert relative_error <= MAX_RELATIVE_COEFFICIENT_ERROR, (
        f"Reconstructed radial velocity profile's fitted Poiseuille coefficient "
        f"({fitted_coefficient:.6f}) differs from the theoretical one "
        f"({theoretical_coefficient:.6f}) by {relative_error:.1%}, more than the allowed "
        f"{MAX_RELATIVE_COEFFICIENT_ERROR:.0%}. See {save_path} to inspect the profile."
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
