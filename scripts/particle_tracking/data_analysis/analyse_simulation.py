"""Validate CTracks/TrackPy particle tracking against a simulated ground truth: match detections
to ground-truth tracks, compute precision/recall/error statistics, and render the paper's
comparison plots.
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.particle_tracking.data_analysis.analysis_functions import plot_velocity_distributions_general, \
    plot_normalized_step_lines, find_matches, calc_stats, plot_abs_velocity_errors, plot_angle_err, \
    plot_velocity_err_dist, load_ctrack_files
from scripts.particle_tracking.data_analysis.vis3D.plot_3d_vectors import plot_single


# ============================================================================
# User-adjustable configuration
# ============================================================================

# --- Acquisition parameters (simulated ground truth) ----------------------------
ACQUISITION_PARAMS = {
    "proj_per_frame": 851,  # simulated sub-steps per recon frame
    "num_frames": 15,
    "start_frame": 1,
}
ACQUISITION_PARAMS["end_frame"] = ACQUISITION_PARAMS["num_frames"]

# File type for this script's own 2D (matplotlib) figures - "png" or "svg".
# Does not affect the 3D pyvista vector plots (*_vector_plot.png), which are always png.
FIGURE_EXT = "png"

# --- Data root - change this if CTracksSubmissionData is moved or mounted under a
# different drive/path (e.g. moving to a new PC). ---------------------------------
DATA_ROOT = "J:\\"

# --- Per-simulation datasets (pick one) --------------------------------------------------
_POREVISCO_BASEFOLDER = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Simulated")

DATASETS = {
    "porevisco_27x": dict(
        basefolder=_POREVISCO_BASEFOLDER,
        ctracks_dir=os.path.join(_POREVISCO_BASEFOLDER, "ctracks_trajectories"),
        velocity_file=os.path.join(_POREVISCO_BASEFOLDER, "rdl_results", "simulated_27x__velocityPoints.csv"),
        mask_file=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Simulation\fields\mask.tif"),
        savefolder=os.path.join(_POREVISCO_BASEFOLDER, "analysis_results"),
    ),
}
ACTIVE_DATASET = "porevisco_27x"
cfg = DATASETS[ACTIVE_DATASET]
basefolder = cfg['basefolder']
ctracks_dir = cfg['ctracks_dir']
velocity_file = cfg['velocity_file']
mask_file = cfg['mask_file']
savefolder = cfg['savefolder']

os.makedirs(savefolder, exist_ok=True)

# --- Matching / analysis parameters ---------------------------------------------------------
MAX_VELOCITY = 60  # x-axis cap (vox/frame) for the recall-vs-velocity plot
VELOCITY_FIELD_RANGE = (3, 25)  # unused by the current script flow; kept for parity with the original config
PLOT_TITLE = "27X"  # unused by the current script flow; kept for parity with the original config
EXPOSURE_TIME_MS = 35.25  # unused by the current script flow; kept for parity with the original config

DIST_TOL = 1.73  # particle-matching distance tolerance (voxels)
VELOCITY_DIST_NUM_BINS = 20
VELOCITY_HIST_BINS = 70
RECALL_THRESHOLD_PCT = 20  # recall percentage used to find the detection-rate velocity cutoff
ERROR_CI_PERCENTILE = 90
LINEARITY_MIN_PATH_LENGTH = 0.5
CONTINUITY_VELOCITY_MAX = 10  # vox/frame cap used when analysing frame-to-frame detection continuity

# --- 3D vector plot styling -----------------------------------------------------------------
DEFAULT_PLOT_PARAMS = {
    "scale": 1,
    "down_sample_factor": 4,
    "crop_bounds": None,
    "arrow_lim": (0.5, 1),
    "plot_grains": True,
    "show_arrowhead": True,
    "surface_transparency": 0.2,
    "clim": (1, 20),
    "velocity_split": 5,
    "plot_low": True,
    "interactive": False,
}
CAM_PARAMS = {
    "azimuth": 90,
    "elevation": 15,
    "focal_shift": [-50, 0, -30],
    "zoom": 1.4,
}
CBAR_PARAMS = {
    "title": "Velocity magnitude\n(vox/scan)",
    "height": 0.35,
    "vertical": True,
    "position_x": 0.05,
    "position_y": 0.35,
    "n_labels": 3,
    "fmt": "%.0f",
    "title_font_size": 26,
    "label_font_size": 26,
    "font_family": "arial",
}

plt.rcParams['font.size'] = 18


# ============================================================================
# Implementation
# ============================================================================

def plot_detectability_vs_linearity(tracks, detected, min_path_length=0.0, savefile = "linearity.svg"):
    """Plot detection recall and mean velocity as a function of ground-truth track linearity.

    `tracks` has shape (N particles, F frames, M control points, 3). `detected` is a list of
    length F containing, for each frame, the ground-truth indices that were matched/detected.
    Tracks with total path length below `min_path_length` are excluded.
    """
    N, F, M, _ = tracks.shape

    # 1. Build and flatten the detection mask
    is_detected = np.zeros((N, F), dtype=bool)
    for f, indices in enumerate(detected):
        if len(indices) > 0:
            is_detected[indices, f] = True
    detected_flat = is_detected.reshape(-1)

    # 2. Flatten the tracks array and calculate Kinematics
    tracks_flat = tracks.reshape(N * F, M, 3)

    start_points = tracks_flat[:, 0, :]
    end_points = tracks_flat[:, -1, :]
    end_to_end_distance = np.linalg.norm(end_points - start_points, axis=1)

    step_vectors = np.diff(tracks_flat, axis=1)
    step_lengths = np.linalg.norm(step_vectors, axis=2)
    total_path_length = np.sum(step_lengths, axis=1)
    mean_velocity = np.mean(step_lengths, axis=1) * M

    with np.errstate(divide='ignore', invalid='ignore'):
        linearity = end_to_end_distance / total_path_length
        linearity = np.where(total_path_length == 0, 1.0, linearity)

    # 3. Filter by minimum path length
    valid_mask = total_path_length >= min_path_length
    linearity = linearity[valid_mask]
    detected_flat = detected_flat[valid_mask]
    mean_velocity = mean_velocity[valid_mask]

    if len(linearity) == 0:
        print(f"No tracks found with a total path length >= {min_path_length}")
        return
    # 4. Bin data
    bins = np.linspace(0, 1, 41) # 40 bins from 0 to 1
    hist_total, _ = np.histogram(linearity, bins=bins)
    hist_detected, _ = np.histogram(linearity[detected_flat], bins=bins)

    # Calculate fraction safely
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction_detected = np.where(hist_total > 0, hist_detected / hist_total, np.nan)

    # Mean velocity per linearity bin
    hist_velocity_sum, _ = np.histogram(linearity, bins=bins, weights=mean_velocity)
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_mean_velocity = np.where(hist_total > 0, hist_velocity_sum / hist_total, np.nan)

    # Setup the plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Primary Y-Axis: Fraction Detected (Blue)
    color_detected = '#1f77b4' # Standard clean blue
    ax.stairs(fraction_detected * 100, bins, color=color_detected, linewidth=2, baseline=None)
    ax.set_xlabel('Linearity (Displacement / Path length)')
    ax.set_ylabel('Recall (%)', color=color_detected)
    ax.set_ylim(0, 70) # Adjusted from your snippet to ensure full 0-100% view

    # Secondary Y-Axis: Mean Velocity (Orange/Red)
    ax2 = ax.twinx()
    color_velocity = '#ff7f0e' # Standard clean orange to contrast blue
    ax2.stairs(bin_mean_velocity, bins, color=color_velocity, linewidth=2, baseline=None)
    ax2.set_ylabel('Mean particle velocity\n(vox/frame)', color=color_velocity)
    ax2.set_ylim(0, 60)

    # Axis Limits (Shared X-axis limits apply to both automatically)
    ax.set_xlim(1.0, 0.8)  # Inverted X-axis limits
    x_ticks = [0.8, 0.85, 0.9, 0.95, 1.0]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['0.8', '0.85', '0.9', '0.95', '1.0'])

    ax.tick_params(axis='x', pad=12)

    plt.tight_layout()
    plt.savefig(savefile)
    plt.show()


def plot_detection_continuity(N, F, detected, velocity_max = None, velocities = None):
    """
    Calculates frame-to-frame conditional detection probabilities
    and plots the Markov transition matrix alongside tracklet survival.

    Parameters:
    - N: int, Total number of ground truth particles
    - F: int, Total number of frames
    - detected: list of length F containing 1D ndarrays of detected indices
    """
    velocities = np.linalg.norm(velocities, axis=-1)
    if velocity_max is not None and velocities is not None:
        global_valid_mask = np.ones(N, dtype=bool)
        for f in range(F):
            global_valid_mask &= (np.asarray(velocities[f]) <= velocity_max)

        # Get the new total of allowed particles
        N_allowed = np.sum(global_valid_mask)

        # Create a mapping from the old particle IDs to the new compressed IDs
        # e.g., if valid IDs are [0, 2, 5], they become [0, 1, 2]
        old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(np.where(global_valid_mask)[0])}
    else:
        N_allowed = N
        old_to_new_id = None
    is_detected = np.zeros((N_allowed, F), dtype=bool)

    for f, indices in enumerate(detected):
        if len(indices) > 0:
            if old_to_new_id is not None:
                # Only keep indices that survived the global velocity filter
                valid_indices = [idx for idx in indices if idx in old_to_new_id]
                # Map them to their new row positions in the smaller matrix
                mapped_indices = [old_to_new_id[idx] for idx in valid_indices]

                if mapped_indices:
                    is_detected[mapped_indices, f] = True
            else:
                is_detected[indices, f] = True

    # 2. Calculate Markov Transition Probabilities
    # Compare state at time t to state at time t+1
    current_state = is_detected[:, :-1]
    next_state = is_detected[:, 1:]

    # Count the raw transitions
    D_to_D = np.sum((current_state == True) & (next_state == True))
    D_to_M = np.sum((current_state == True) & (next_state == False))
    M_to_D = np.sum((current_state == False) & (next_state == True))
    M_to_M = np.sum((current_state == False) & (next_state == False))

    # Calculate conditional probabilities (rows sum to 1.0)
    # P(Next | Current)
    P_D_given_D = D_to_D / (D_to_D + D_to_M) if (D_to_D + D_to_M) > 0 else 0
    P_M_given_D = D_to_M / (D_to_D + D_to_M) if (D_to_D + D_to_M) > 0 else 0
    P_D_given_M = M_to_D / (M_to_D + M_to_M) if (M_to_D + M_to_M) > 0 else 0
    P_M_given_M = M_to_M / (M_to_D + M_to_M) if (M_to_D + M_to_M) > 0 else 0

    transition_matrix = np.array([
        [P_D_given_D, P_M_given_D],
        [P_D_given_M, P_M_given_M]
    ])

    # 3. Calculate Continuous Run-Lengths (Tracklet Lengths)
    # Pad with False at the start and end of the F dimension to cleanly catch borders
    padded = np.pad(is_detected, ((0, 0), (1, 1)), mode='constant', constant_values=False)
    # np.diff finds the edges: +1 means changed to True, -1 means changed to False
    diffs = np.diff(padded.astype(int), axis=1)

    starts = np.where(diffs == 1)
    ends = np.where(diffs == -1)
    # The length of each continuous streak of True values
    run_lengths = ends[1] - starts[1]

    if len(run_lengths) == 0:
        print("No continuous tracks found.")
        return None, None

    max_run = np.max(run_lengths)

    # 4. Publication-Ready Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # --- Panel A: Transition Probability Matrix Heatmap ---
    cax = ax1.matshow(transition_matrix, cmap='Blues', vmin=0, vmax=1)

    # Overlay the text values
    for i in range(2):
        for j in range(2):
            text_color = "white" if transition_matrix[i, j] > 0.5 else "black"
            ax1.text(j, i, f"{transition_matrix[i, j]:.3f}",
                     ha="center", va="center", color=text_color, fontsize=14, weight='bold')

    ax1.set_title('Frame-to-Frame Transition Matrix', fontsize=14, pad=15)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Detected (t+1)', 'Missed (t+1)'], fontsize=12)
    ax1.set_yticklabels(['Detected (t)', 'Missed (t)'], fontsize=12)
    ax1.tick_params(axis='both', which='both', length=0)  # Hide ticks for cleaner matrix

    # --- Panel B: Tracklet Survival Curve (Reverse Cumulative) ---
    # Bins from 1 to max_run + 1
    bins = np.arange(1, max_run + 2)

    # cumulative=-1 plots P(X >= x), log=True shows the exponential decay characteristic of tracking fragmentation
    ax2.hist(run_lengths, bins=bins, cumulative=-1, density=True,
             histtype='step', color='black', linewidth=2)

    ax2.set_title('Tracklet Survival Curve', fontsize=14)
    ax2.set_xlabel('Continuous Detection Streak (Frames)', fontsize=13)
    ax2.set_ylabel('Probability of Streak $\geq$ X', fontsize=13)
    ax2.set_yscale('log')

    # Clean up aesthetics
    ax2.tick_params(axis='both', which='major', labelsize=11, direction='in', length=6, width=1.5)
    ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.show()

def load_groundtruth(proj_per_frame = 850, num_frames = 7, start_frame = 1):
    """Load the simulated ground-truth tracks and derive per-frame positions/velocities.

    Ground truth is stored as `proj_per_frame` simulated sub-steps per frame; positions and
    velocities are averaged over each frame's sub-steps.

    Returns:
        gt_tracks_frames: (num_frames - start_frame, N, 3) per-frame mean position.
        gt_vel_frames: (num_frames - start_frame, N, 3) per-frame mean velocity.
        gt_tracks: (N, num_frames - start_frame, proj_per_frame, 3) full-resolution tracks.
    """
    gt_tracks = np.load(os.path.join(basefolder, "ground_truth_tracks.npy"), allow_pickle=True)  # N,851*7,3
    gt_vel = np.gradient(gt_tracks, 1 / proj_per_frame, axis=1, edge_order=1)
    gt_tracks = gt_tracks.reshape(-1, num_frames, proj_per_frame, 3) #N,7,851,3
    gt_tracks_frames = gt_tracks.mean(axis=2)  # N,7,3
    gt_vel_frames = gt_vel.reshape(-1, num_frames, proj_per_frame, 3).mean(axis=2)  # N,7,3

    gt_tracks_frames = np.swapaxes(gt_tracks_frames, 0, 1) #7,N,3
    gt_vel_frames = np.swapaxes(gt_vel_frames, 0, 1) #7,N,3

    return gt_tracks_frames[start_frame:], gt_vel_frames[start_frame:], gt_tracks[:,start_frame:,:,:]

def load_ctracks(start_frame = 1, end_frame=7):
    """Load per-frame CTracks reconstruction files and reduce each to a position and velocity.

    Thin wrapper around the shared `load_ctrack_files` loader (see `analysis_functions.py`), which
    implements the same scan-file -> position/velocity reduction used by the other analyse_*.py scripts.

    Returns:
        (ctrack_pos_analysis, ctrack_vel_analysis): per-frame lists of position/velocity arrays;
        the number of particles N can differ per frame.
    """
    files = [os.path.join(ctracks_dir, f"params_rot{i}.npy") for i in range(start_frame, end_frame)]
    ctrack_pos_analysis, ctrack_vel_analysis, _, _, _, _ = load_ctrack_files(files)
    return ctrack_pos_analysis, ctrack_vel_analysis #6,<N>,3 - N changes per frame

def load_trackpy(start_frame = 1, end_frame=7):
    """Load per-frame TrackPy/RDL detections from the results CSV.

    Returns:
        (trackpy_pos_analysis, trackpy_vel_analysis): per-frame lists of position/velocity arrays.
    """
    tp_data = pd.read_csv(velocity_file)  # dataframe particle, frame, z,y,x,vz,vy,vx,velMags
    trackpy_pos_analysis, trackpy_vel_analysis = [], []
    for frame in range(start_frame, end_frame):
        trackpy_analysis = tp_data[tp_data['frame'] == frame]
        trackpy_pos_analysis.append(trackpy_analysis[['x', 'y', 'z']].values)
        trackpy_vel_analysis.append(trackpy_analysis[['vx', 'vy', 'vz']].values)
    return trackpy_pos_analysis, trackpy_vel_analysis #6,<N>,3


def make_and_save_dataframe(positions, velocities, savefile):
    """Flatten per-frame position/velocity arrays into a single frame-0 dataframe and save as CSV."""
    def standardize(data):
        """Concatenate a (possibly ragged) list/tuple of arrays into one (M, 3) array."""
        if not isinstance(data, (list, tuple)):
            data = [data]
        # Reshape each piece individually to handle ragged arrays, then stack
        return np.concatenate([np.asarray(chunk).reshape(-1, 3) for chunk in data], axis=0)

    # --- 1. Standardize Inputs ---
    pos_flat = standardize(positions)
    vel_flat = standardize(velocities)

    combined_data = np.hstack((pos_flat, vel_flat))
    df = pd.DataFrame(combined_data, columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
    df.insert(0, 'frame', 0)
    df.to_csv(savefile, index=False)


def load_all_tracks():
    """Load ground-truth, CTracks, and TrackPy tracks for the active dataset.

    Returns:
        gt_pos, gt_vel, gt_alltracks, ctracks_pos, ctracks_vel, trackpy_pos, trackpy_vel.
        Each of gt_pos/gt_vel/ctracks_pos/ctracks_vel/trackpy_pos/trackpy_vel is a per-frame list
        (particle count can differ per frame for CTracks/TrackPy); gt_alltracks has shape
        (N, frames, sub-steps, 3).
    """
    gt_pos, gt_vel, gt_alltracks = load_groundtruth(start_frame=ACQUISITION_PARAMS['start_frame'],
                                                     proj_per_frame=ACQUISITION_PARAMS['proj_per_frame'],
                                                     num_frames=ACQUISITION_PARAMS['num_frames'])  # gt_alltracks shape N,7,851,3
    ctracks_pos, ctracks_vel = load_ctracks(start_frame=ACQUISITION_PARAMS['start_frame'],
                                            end_frame=ACQUISITION_PARAMS['end_frame'])
    trackpy_pos, trackpy_vel = load_trackpy(start_frame=ACQUISITION_PARAMS['start_frame'],
                                            end_frame=ACQUISITION_PARAMS['end_frame'])
    return gt_pos, gt_vel, gt_alltracks, ctracks_pos, ctracks_vel, trackpy_pos, trackpy_vel


def plot_velocity_distribution_comparison(gt_vel, ctracks_vel, trackpy_vel):
    """Plot combined velocity-magnitude distributions for ground truth, CTracks, and TrackPy.

    Returns:
        (n_particles_list, velocities_list_mag): total detected-particle counts and velocity
        magnitude arrays, each ordered [ground truth, CTracks, TrackPy].
    """
    velocities_list = [
        np.concatenate(gt_vel, axis=0),
        np.concatenate(ctracks_vel, axis=0),
        np.concatenate(trackpy_vel, axis=0),
    ]
    velocities_list_mag = [np.linalg.norm(velocities_list[i], axis=1) for i in range(3)]
    n_particles_list = [len(velocities_list_mag[i]) for i in range(3)]

    plot_velocity_distributions_general(velocities_list, ['Ground truth', 'CTracks', 'RDL'], num_bins=VELOCITY_DIST_NUM_BINS,
                                        save_file=os.path.join(savefolder, f"velocity_distribution.{FIGURE_EXT}"),
                                        colors = ['green', 'blue', 'red'], log_y = True)

    return n_particles_list, velocities_list_mag


def match_and_compute_detection_stats(gt_pos, gt_vel, ctracks_pos, ctracks_vel, trackpy_pos, trackpy_vel, n_particles_list):
    """Match ground-truth tracks to CTracks/TrackPy detections and print precision/recall/F1.

    Returns a dict of the matched ground-truth/reconstructed positions, velocities, velocity
    magnitudes, and per-frame match indices for both CTracks and TrackPy.
    """
    (gt_ctracks_true_pos, gt_ctracks_true_vel, gt_ctracks_match_indices), (ctracks_true_pos, ctracks_true_vel, ctracks_match_indices) = \
        find_matches(gt_pos, gt_vel, ctracks_pos, ctracks_vel, tolerance=DIST_TOL) #shape 6*N,3
    (gt_trackpy_true_pos, gt_trackpy_true_vel, gt_trackpy_match_indices), (trackpy_true_pos, trackpy_true_vel, trackpy_match_indices) = \
        find_matches(gt_pos, gt_vel, trackpy_pos, trackpy_vel, tolerance=DIST_TOL) #shape 6*N,3

    # magnitudes of true detections
    gt_ctracks_true_vel_mag = np.linalg.norm(gt_ctracks_true_vel, axis=-1)
    gt_trackpy_true_vel_mag = np.linalg.norm(gt_trackpy_true_vel, axis=-1)
    ctracks_true_vel_mag = np.linalg.norm(ctracks_true_vel, axis=-1)
    trackpy_true_vel_mag = np.linalg.norm(trackpy_true_vel, axis=-1)

    # detection rates - TP, FP, F1
    ctracks_stats = calc_stats(ctracks_true_pos.shape[0], n_particles_list[0], n_particles_list[1])
    trackpy_stats = calc_stats(trackpy_true_pos.shape[0], n_particles_list[0], n_particles_list[2])
    print(f"CTracks precision: {ctracks_stats[0]:.1%}\tRecall:{ctracks_stats[1]:.1%}\tF1:{ctracks_stats[2]:.1%}")
    print(f"TrackPy precision: {trackpy_stats[0]:.1%}\tRecall:{trackpy_stats[1]:.1%}\tF1:{trackpy_stats[2]:.1%}")

    return dict(
        gt_ctracks_true_pos=gt_ctracks_true_pos, gt_ctracks_true_vel=gt_ctracks_true_vel,
        gt_ctracks_match_indices=gt_ctracks_match_indices,
        ctracks_true_pos=ctracks_true_pos, ctracks_true_vel=ctracks_true_vel, ctracks_match_indices=ctracks_match_indices,
        gt_trackpy_true_pos=gt_trackpy_true_pos, gt_trackpy_true_vel=gt_trackpy_true_vel,
        gt_trackpy_match_indices=gt_trackpy_match_indices,
        trackpy_true_pos=trackpy_true_pos, trackpy_true_vel=trackpy_true_vel, trackpy_match_indices=trackpy_match_indices,
        gt_ctracks_true_vel_mag=gt_ctracks_true_vel_mag, gt_trackpy_true_vel_mag=gt_trackpy_true_vel_mag,
        ctracks_true_vel_mag=ctracks_true_vel_mag, trackpy_true_vel_mag=trackpy_true_vel_mag,
    )


def plot_detection_and_error_analysis(velocities_list_mag, matches):
    """Plot detection-recall-vs-velocity, absolute/percentage velocity error, and angle error comparisons."""
    plot_normalized_step_lines(velocities_list_mag[0], matches['ctracks_true_vel_mag'], matches['trackpy_true_vel_mag'],
                               xlabel = "Velocity magnitude (vox/frame)",
                               log = True, bins=VELOCITY_HIST_BINS, normalize=False,
                               filename = os.path.join(savefolder, f"velocity_histogram.{FIGURE_EXT}"), max_velocity=MAX_VELOCITY,
                               data2_label="RDL", threshold_pct=RECALL_THRESHOLD_PCT, plot_recall=True)

    # velocity errors - median with confidence interval
    abs_err_ctracks = matches['ctracks_true_vel_mag'] - matches['gt_ctracks_true_vel_mag']
    print("Median CTracks error:", np.median(abs_err_ctracks))
    print(f"{ERROR_CI_PERCENTILE}% confidence", np.percentile(abs_err_ctracks, ERROR_CI_PERCENTILE))

    # absolute velocity error vs magnitude, with relative-error fan lines
    plot_abs_velocity_errors((matches['gt_ctracks_true_vel_mag'], matches['ctracks_true_vel_mag']),
                                      (matches['gt_trackpy_true_vel_mag'], matches['trackpy_true_vel_mag']),
                                      save_file = os.path.join(savefolder, f"abs_velocity_error.{FIGURE_EXT}"),
                             plot_trackpy=True, ax = None)

    # velocity error angle vs magnitude (normalised: err max = arcsin(2*DIST_TOL/v))
    plot_angle_err((matches['gt_ctracks_true_vel'], matches['ctracks_true_vel']),
                    (matches['gt_trackpy_true_vel'], matches['trackpy_true_vel']),
                    os.path.join(savefolder, f"error_angle.{FIGURE_EXT}"), normalise = True, plot_trackpy = True, ax = None)

    # velocity error percent distribution
    plot_velocity_err_dist((matches['gt_ctracks_true_vel_mag'], matches['ctracks_true_vel_mag']),
                    (matches['gt_trackpy_true_vel_mag'], matches['trackpy_true_vel_mag']),
                           use_percentage=True, plot_trackpy=True, savefile = os.path.join(savefolder, f"abs_velocity_error_dist.{FIGURE_EXT}"))


def plot_linearity_and_continuity_analysis(gt_alltracks, matches, gt_vel):
    """Plot CTracks detectability vs. track linearity, and frame-to-frame detection continuity."""
    plot_detectability_vs_linearity(gt_alltracks, matches['gt_ctracks_match_indices'],
                                    min_path_length=LINEARITY_MIN_PATH_LENGTH,
                                    savefile = os.path.join(savefolder, f"linearity_detectability.{FIGURE_EXT}"))

    # TODO filter to low velocities and linear tracks only
    plot_detection_continuity(gt_alltracks.shape[0], gt_alltracks.shape[1], matches['gt_ctracks_match_indices'],
                              velocity_max = CONTINUITY_VELOCITY_MAX, velocities = gt_vel)


def save_and_plot_vector_csvs(gt_pos, gt_vel, matches):
    """Export ground-truth/CTracks/TrackPy true-positive tracks to CSV and render 3D vector plots."""
    savefiles = ['ground_truth', 'ctracks', 'trackpy']
    plot_data = [
        [gt_pos, gt_vel],
        [matches['ctracks_true_pos'], matches['ctracks_true_vel']],
        [matches['trackpy_true_pos'], matches['trackpy_true_vel']],
    ]
    for i in range(3): #TODO find good crop
        df_file = os.path.join(savefolder,savefiles[i]+"_plotting_vectors.csv")
        make_and_save_dataframe(*plot_data[i],df_file)
        save_file = os.path.join(savefolder,savefiles[i] + "_vector_plot.png")
        plot_single(mask_file, df_file, save_file = save_file, cam_params = CAM_PARAMS, cbar_params=CBAR_PARAMS, **DEFAULT_PLOT_PARAMS)

    # TODO: plot ground truth vectors, colored by whether they were detected
    # TODO: plot interpolated ctracks/trackpy/gt fields - difference images?


def main():
    """Validate CTracks/TrackPy against simulated ground truth: matching, error stats, and visualizations."""
    gt_pos, gt_vel, gt_alltracks, ctracks_pos, ctracks_vel, trackpy_pos, trackpy_vel = load_all_tracks()
    n_particles_list, velocities_list_mag = plot_velocity_distribution_comparison(gt_vel, ctracks_vel, trackpy_vel)
    matches = match_and_compute_detection_stats(gt_pos, gt_vel, ctracks_pos, ctracks_vel, trackpy_pos, trackpy_vel,
                                                n_particles_list)
    plot_detection_and_error_analysis(velocities_list_mag, matches)
    plot_linearity_and_continuity_analysis(gt_alltracks, matches, gt_vel)
    save_and_plot_vector_csvs(gt_pos, gt_vel, matches)


if __name__ == "__main__":
    main()
