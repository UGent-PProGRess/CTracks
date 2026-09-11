"""Shared analysis/plotting utilities used by the `analyse_capillary.py`, `analyse_fracture.py`,
`analyse_porous.py`, and `analyse_simulation.py` scripts in this package.

Roughly grouped by purpose:
- Matching/scoring: `find_matches`, `test_endpoints`, `calc_f1`, `calc_stats` - match ground-truth
  and reconstructed tracks by endpoint distance and score the match (precision/recall/F1).
- Loading: `load_ctrack_files` - reads per-frame CTrack `.npy` reconstruction results into
  position/velocity/radius/attenuation arrays (see its docstring for a note on the separate,
  intentionally-not-deduplicated `vis3D/utils/utils.py::convert_np_to_df`, which does the same
  extraction for a different (raw-file) calling context).
- Plotting: velocity-distribution histograms (`plot_velocity_distributions*`), masked
  velocity-field slice images (`plot_two_velocity_slices`, `plot_velocity_field_slice`,
  sharing `default_velocity_cmap`), binned error-vs-magnitude curves (`plot_abs_velocity_errors`,
  `plot_angle_err`, sharing the private `_binned_mean_step_line` helper), and error-distribution
  histograms (`plot_velocity_err_dist`).
- Metrics: `calculate_and_print_metrics_comparison`.

Each `analyse_*.py` script imports only the subset of these it needs; see each script's own
imports for exactly which functions are live for that dataset.
"""
import os

import scipy as sp
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import scipy.ndimage as spnd
import scipy.interpolate as spint
import tifffile
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.spatial import KDTree
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm, gaussian_kde

def find_matches(gt_pos, gt_vel, recon_pos, recon_vel, tolerance = 1.73):
    """Match ground-truth and reconstructed particles frame-by-frame via endpoint distance.

    For each frame, builds start/end track endpoints from position and velocity and calls
    `test_endpoints` to find a one-to-one nearest-neighbor matching within `tolerance`.

    Returns:
        ((matched_gt_pos, matched_gt_vel, per_frame_gt_indices),
         (matched_recon_pos, matched_recon_vel, per_frame_recon_indices))
        where the position/velocity arrays are stacked across all frames.
    """
    assert len(gt_pos) == len(recon_pos), "Num frames wrong"
    matched_gt_pos, matched_gt_vel, matched_recon_pos, matched_recon_vel = [], [], [], []
    match_gt_indices, match_recon_indices = [], []
    for i in range(len(gt_pos)): #for each frame
        frame_gt_pos, frame_gt_vel, frame_recon_pos, frame_recon_vel = gt_pos[i], gt_vel[i], recon_pos[i], recon_vel[i]
        gt_endpoints = np.stack([frame_gt_pos - frame_gt_vel/2, frame_gt_pos + frame_gt_vel/2], axis=1)
        recon_endpoints = np.stack([frame_recon_pos - frame_recon_vel / 2, frame_recon_pos + frame_recon_vel / 2], axis=1)
        num_particles, _, (matched_gt_indices, matched_recon_indices) = test_endpoints(gt_endpoints, recon_endpoints, tolerance)
        matched_gt_pos.append(frame_gt_pos[matched_gt_indices])
        matched_gt_vel.append(frame_gt_vel[matched_gt_indices])
        matched_recon_pos.append(frame_recon_pos[matched_recon_indices])
        matched_recon_vel.append(frame_recon_vel[matched_recon_indices])
        match_gt_indices.append((matched_gt_indices))
        match_recon_indices.append((matched_recon_indices))
    return (np.vstack(matched_gt_pos), np.vstack(matched_gt_vel), match_gt_indices),(np.vstack(matched_recon_pos), np.vstack(matched_recon_vel), match_recon_indices)

def calc_f1(num_true_particles, num_gt, num_recon):
    """Compute the F1 score from a true-positive count and the ground-truth/reconstructed totals."""
    precision = num_true_particles / num_recon
    recall = num_true_particles / num_gt
    if precision == 0 and recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calc_stats(num_true_particles, num_gt, num_recon):
    """Compute precision, recall, and F1 score from a true-positive count and the totals."""
    precision = num_true_particles/num_recon
    recall = num_true_particles/num_gt
    f1 = calc_f1(num_true_particles, num_gt, num_recon)
    return precision, recall, f1

def test_endpoints(gt_tracks, recon_tracks, min_truth_distance = 1.73):
    """Greedily match ground-truth and reconstructed tracks by summed start/end-point distance.

    Computes pairwise distances between each ground-truth and reconstructed track's start and
    end points, then greedily assigns matches from closest to farthest pair (one-to-one), stopping
    once the summed distance exceeds `2 * min_truth_distance`.

    Returns:
        ((num_true_particles, num_gt, num_recon),
         (matched_gt_points, gt_unmatched_points, recon_unmatched_points),
         (matched_gt_indices, matched_recon_indices))
    """
    num_gt = gt_tracks.shape[0]
    num_recon = recon_tracks.shape[0]

    # Extract only the start (0) and end (-1) points
    # gt_start_end: (num_gt, 2, 3)
    gt_start_end = gt_tracks[:, [0, -1], :]
    # recon_start_end: (num_recon, 2, 3)
    recon_start_end = recon_tracks[:, [0, -1], :]

    # --- 1. Pairwise Distance Calculation ---

    # Expand for pairwise distance computation:
    # (N_gt, 1, 2, 3) vs (1, N_recon, 2, 3)
    # Using np.newaxis is equivalent to torch.unsqueeze
    gt_exp = gt_start_end[:, np.newaxis, :, :]
    recon_exp = recon_start_end[np.newaxis, :, :, :]

    # diffs: (num_gt, num_recon, 2, 3)
    diffs = gt_exp - recon_exp

    # dists: (num_gt, num_recon, 2)
    # np.linalg.norm(..., axis=-1) is equivalent to torch.norm(..., dim=-1)
    dists = np.linalg.norm(diffs, axis=-1)

    # Sum of distances for start point and end point
    # total_dists: (num_gt, num_recon)
    total_dists = dists.sum(axis=-1)

    # --- 2. Sorting and Index Mapping ---

    # Flatten all distances and sort the flattened indices (equivalent to torch.sort)
    # all_pairs: (num_gt * num_recon,)
    all_pairs = total_dists.flatten()

    # sorted_indices contains the flattened indices that would sort the array
    sorted_indices = np.argsort(all_pairs)
    # sorted_dists: The actual sorted distances
    sorted_dists = all_pairs[sorted_indices]

    # Map the flattened index back to (gt_idx, recon_idx)
    # Using np.divmod or // and %
    gt_indices = sorted_indices // num_recon
    recon_indices = sorted_indices % num_recon

    # --- 3. Exclusive Matching and Index Collection ---

    # Boolean masks for marking tracks as already matched (one-to-one mapping)
    # Use np.bool_ or bool for boolean dtype
    matched_gt = np.zeros(num_gt, dtype=bool)
    matched_recon = np.zeros(num_recon, dtype=bool)

    matched_gt_indices= []
    matched_recon_indices = []
    num_true_particles = 0

    # Iterate through the pairs from closest distance to farthest
    # Use standard iteration over NumPy arrays
    for gt_idx, recon_idx, dist in zip(gt_indices, recon_indices, sorted_dists):

        # Distance threshold check
        # This is equivalent to the original torch.Tensor.item() logic
        # as NumPy indices and values are already standard Python types.
        if dist > 2 * min_truth_distance:
            break  # Optimization: No need to check further — too far

        # Exclusive match check (Hungarian-like)
        if not matched_gt[gt_idx] and not matched_recon[recon_idx]:
            # Lock the pair
            matched_gt[gt_idx] = True
            matched_recon[recon_idx] = True
            num_true_particles += 1

            # Store the indices of the successfully locked pair
            matched_gt_indices.append(gt_idx)
            matched_recon_indices.append(recon_idx)

    # --- 4. Slicing to Achieve Parallelism ---

    # Convert index lists to NumPy arrays for efficient slicing
    # Use np.intp (or np.int64) for indexing
    matched_gt_np_indices = np.array(matched_gt_indices, dtype=np.intp)
    matched_recon_np_indices = np.array(matched_recon_indices, dtype=np.intp)

    # Use the ALIGNED index arrays to slice the start/end points.
    matched_gt_points = gt_start_end[matched_gt_np_indices]
    matched_recon_points = recon_start_end[matched_recon_np_indices]

    # Calculate unmatched points using the boolean masks (boolean indexing)
    gt_unmatched_points = gt_start_end[~matched_gt]
    recon_unmatched_points = recon_start_end[~matched_recon]

    # --- 5. Return Results in Requested Format ---

    return (num_true_particles, num_gt, num_recon), \
        (matched_gt_points, gt_unmatched_points, recon_unmatched_points), \
        (matched_gt_np_indices, matched_recon_np_indices)

def calc_mean_distances(positions):
    """Print the mean nearest-neighbor distance between particles, computed two ways.

    `positions` is a list of per-frame position arrays. Approach A averages the within-frame
    nearest-neighbor distance across frames; Approach B pools all particles (across all frames)
    into one KD-tree and computes a single global nearest-neighbor mean.
    """
    # --- Approach A: Per-Frame Calculations ---
    frame_nn_means = []

    for frame in positions:
        if len(frame) < 2:
            continue
        tree = KDTree(frame)
        distances, _ = tree.query(frame, k=2)
        nn_distances = distances[:, 1]
        frame_nn_means.append(np.mean(nn_distances))

    mean_of_means_nn = np.mean(frame_nn_means) if frame_nn_means else 0.0

    # --- Approach B: Global Pool (Cross-Frame NN) ---
    # 1. Flatten all frames into a single global array of positions
    global_positions = np.vstack(positions)

    # 2. Build a single KD-Tree for ALL particles across ALL time
    global_tree = KDTree(global_positions)

    # 3. Query the global tree.
    # k=2 because the closest neighbor to any point is still its exact identity (distance = 0)
    global_distances, _ = global_tree.query(global_positions, k=2)

    # 4. Extract the second column (the true nearest neighbor in space/time)
    global_nn_distances = global_distances[:, 1]

    # 5. Calculate the global pool mean
    global_pool_mean_nn = np.mean(global_nn_distances)

    # --- Output ---
    print(f"Approach A (Within-Frame Mean of Means): {mean_of_means_nn:.6f}")
    print(f"Approach B (Cross-Frame Global Pool):    {global_pool_mean_nn:.6f}")

def plot_velocity_distributions(ctracks_vel, trackpy_vel_all, num_bins=20, save_file="vel_dist.png"):
    """Plot per-component and magnitude velocity distributions comparing CTracks vs. RDL/TrackPy."""
    ctracks_mag = np.linalg.norm(ctracks_vel, axis=1)
    trackpy_mag = np.linalg.norm(trackpy_vel_all, axis=1)

    # 1. Prepare data and labels for iteration
    data_sets = [
        (ctracks_vel[:, 0], trackpy_vel_all[:, 0]),  # vx
        (ctracks_vel[:, 1], trackpy_vel_all[:, 1]),  # vy
        (ctracks_vel[:, 2], trackpy_vel_all[:, 2]),  # vz
        (ctracks_mag, trackpy_mag)  # |v|
    ]

    xlabels = [
        r'$v_x$ (vx/frame)',
        r'$v_y$ (vx/frame)',
        r'$v_z$ (vx/frame)',
        r'$|\mathbf{v}|$ (vx/frame)'
    ]

    # 2. Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # 3. Iterate and plot
    for i, ax in enumerate(axes):
        data1 = data_sets[i][0]
        data2 = data_sets[i][1]
        xlabel = xlabels[i]

        mean1 = np.mean(data1)
        mean2 = np.mean(data2)

        min_val = min(np.min(data1), np.min(data2))
        max_val = max(np.max(data1), np.max(data2))
        bins = np.linspace(min_val, max_val, num_bins)

        # Plot histograms WITHOUT labels so they don't clutter the subplot legends
        ax.hist(data1, bins=bins, alpha=1, color='b', linewidth=2, histtype='step', density=True)
        ax.hist(data2, bins=bins, alpha=1, color='r', linewidth=2, histtype='step', density=True)

        # Plot mean lines WITH labels (using f-strings for cleaner rounding)
        ax.axvline(mean1, color='b', linestyle='--', label=f"Mean: {mean1:.2f}")
        ax.axvline(mean2, color='r', linestyle='--', label=f"Mean: {mean2:.2f}")

        # Set labels and legend (this legend will now ONLY contain the two mean lines)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Probability Density')
        ax.legend(loc='upper right')

        # 4. Create a Global Figure Legend
    # Define custom proxy artists for the main categories
    custom_lines = [
        Line2D([0], [0], color='b', lw=2),
        Line2D([0], [0], color='r', lw=2)
    ]

    # Place the global legend at the top center of the figure
    fig.legend(custom_lines, ['CTracks', 'RDL'],
               loc='upper right', ncol=2, fontsize=18, frameon=False)

    # 5. Adjust layout and save
    # The rect parameter leaves 5% empty space at the top so the global legend doesn't overlap the plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_file)
    plt.show()

def default_velocity_cmap():
    """Return the default blue-to-red velocity-magnitude colormap shared by the slice-plotting functions
    (`plot_two_velocity_slices`, `plot_velocity_field_slice`, and `analyse_capillary.plot_three_velocity_slices`)."""
    colors = ["blue", "turquoise", "green", "yellow", "orange", "red"]
    return LinearSegmentedColormap.from_list("blue_red", colors)

def plot_two_velocity_slices(mask, velocity_fields, v_range, names, x_crop=0, y_crop = 0, file_loc="./", cmap=None, figsize = (4,12), slice_shift = 0):
    """Plot side-by-side mid-depth velocity-field slices (e.g. two datasets) with a shared colorbar."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout = True)

    if cmap is None:
        cmap = default_velocity_cmap()

    for i, ax in enumerate(axes):
        mid_idx = velocity_fields[i].shape[1] // 2 + slice_shift
        v_slice = velocity_fields[i][:, mid_idx, :]
        m_slice = mask[:, mid_idx, :]

        v_slice = v_slice[y_crop : -y_crop or None, x_crop : -x_crop or None]
        m_slice = m_slice[y_crop : -y_crop or None, x_crop : -x_crop or None]

        grain_mask = (m_slice == 0)

        ax.imshow(grain_mask, cmap='gray', alpha=0.5, vmin=0, vmax=1)

        masked_velocity = np.ma.masked_where(grain_mask, v_slice)
        im = ax.imshow(masked_velocity, cmap=cmap, vmin=v_range[0], vmax=v_range[1])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(names[i])


    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',
                        aspect=20, shrink=0.8, ticks=[v_range[0], v_range[1]], pad = 0.02)
    cbar.set_label('Velocity magnitude\n(voxels/frame)')

    plt.savefig(file_loc, bbox_inches='tight', dpi=300)
    plt.show()

def plot_velocity_field_slice(mask, velocity_field,  v_range = (0,6), file_loc = "./", cmap = None):
    """Plot a single masked velocity-field slice and save a matching standalone colorbar image."""
    fig, ax = plt.subplots()
    grain_mask = (mask == 0)
    ax.imshow(grain_mask, cmap='gray', alpha=0.5, vmin=0, vmax=1)
    if cmap is None:
        cmap = default_velocity_cmap()
    masked_velocity = np.ma.masked_where(grain_mask, velocity_field)
    im = ax.imshow(masked_velocity, cmap=cmap, vmin=v_range[0], vmax = v_range[1])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(file_loc, bbox_inches='tight')
    plt.show()
    plt.close()

    fig_colorbar = plt.figure(figsize=(0.8, 4))  # Adjust size as needed
    ax_colorbar = fig_colorbar.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]

    cbar = fig.colorbar(im, cax=ax_colorbar)
    colorbar_filedir = os.path.dirname(file_loc)
    fig_colorbar.savefig(os.path.join(colorbar_filedir, f"colorbar-{v_range[0]}_{v_range[1]}.png"), bbox_inches='tight')
    plt.close()

def plot_velocity_distributions_general(velocities_list, labels, num_bins=20, save_file="vel_dist.png", colors=None, log_y=False):
    """Plot per-component and magnitude velocity distributions for an arbitrary number of tracking datasets.

    Parameters:
    - velocities_list: List of Nx3 numpy arrays containing (vx, vy, vz).
    - labels: List of strings corresponding to the datasets.
    - num_bins: Number of histogram bins.
    - save_file: Output filename.
    - colors: Optional list of matplotlib color strings.
    - log_y: Boolean to set the y-axis to a logarithmic scale.
    """
    num_datasets = len(velocities_list)

    # Handle colors: Use provided list or default matplotlib cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Repeat color cycle if there are more datasets than default colors
        if num_datasets > len(colors):
            colors = colors * (num_datasets // len(colors) + 1)

    # 1. Calculate magnitudes for all datasets
    magnitudes_list = [np.linalg.norm(vel, axis=1) for vel in velocities_list]

    # 2. Prepare data by grouping components across all datasets
    data_components = [
        [vel[:, 0] for vel in velocities_list],  # vx
        [vel[:, 1] for vel in velocities_list],  # vy
        [vel[:, 2] for vel in velocities_list],  # vz
        magnitudes_list  # |v|
    ]

    xlabels = [
        r'$v_x$ (vx/frame)',
        r'$v_y$ (vx/frame)',
        r'$v_z$ (vx/frame)',
        r'$|\mathbf{v}|$ (vx/frame)'
    ]

    # 3. Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # 4. Iterate and plot
    for i, ax in enumerate(axes):
        comp_data = data_components[i]
        xlabel = xlabels[i]

        # Dynamically determine the global min and max for consistent binning
        min_val = min([np.min(data) for data in comp_data])
        max_val = max([np.max(data) for data in comp_data])
        bins = np.linspace(min_val, max_val, num_bins)

        # Plot histograms and mean lines for each dataset
        for j, data in enumerate(comp_data):
            color = colors[j]
            mean_val = np.mean(data)

            # ADDED: log=log_y cleanly handles the logarithmic scaling
            ax.hist(data, bins=bins, alpha=1, color=color, linewidth=2,
                    histtype='step', density=False, log=log_y)

            # Plot mean lines WITH labels
            ax.axvline(mean_val, color=color, linestyle='--',
                       label=f"Mean: {mean_val:.2f}")

        # Set labels and legend
        ax.set_xlabel(xlabel)
        # Dynamically update the y-label so anyone reading the plot knows it's a log scale
        ylabel = 'Counts (Log Scale)' if log_y else 'Counts'
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right')

    # 5. Create a Global Figure Legend dynamically
    custom_lines = [Line2D([0], [0], color=colors[j], lw=2) for j in range(num_datasets)]

    fig.legend(custom_lines, labels,
               loc='upper right', ncol=min(num_datasets, 4), fontsize=18, frameon=False)

    # 6. Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_file)
    plt.show()


def plot_normalized_step_lines(ground_truth, ctracks_recon, trackpy_recon, xlabel="", bins=50, normalize=True,
                               log=False, data1_label="CTracks", data2_label="TrackPy", filename="vel_hist",
                               max_velocity=35, threshold_pct=None, plot_recall=False):
    """Plot normalized step histograms (or recall curves) comparing ground truth, CTracks, and TrackPy.

    If `threshold_pct` is given, also computes the velocity at which each method's detection rate
    (relative to ground truth) first drops below that percentage, returned as `(drop_vel_ct, drop_vel_tp)`.
    If `plot_recall` is True, plots recall-vs-velocity curves instead of density histograms.
    """
    # --- Data Preparation ---
    plt.rcParams.update({'font.size': 18})
    all_data = np.concatenate([ground_truth, ctracks_recon, trackpy_recon])
    _, common_bins = np.histogram(all_data, bins=bins)

    # Calculate counts for each dataset using the common bins
    counts_gt, _ = np.histogram(ground_truth, bins=common_bins)
    counts_ct, _ = np.histogram(ctracks_recon, bins=common_bins)
    counts_tp, _ = np.histogram(trackpy_recon, bins=common_bins)

    # --- Recall & Threshold Calculation ---
    drop_vel_ct, drop_vel_tp = None, None

    with np.errstate(divide='ignore', invalid='ignore'):
        rate_ct = np.true_divide(counts_ct, counts_gt)
        rate_tp = np.true_divide(counts_tp, counts_gt)

        rate_ct[counts_gt == 0] = np.nan
        rate_tp[counts_gt == 0] = np.nan

    if threshold_pct is not None:
        threshold_frac = threshold_pct / 100.0

        # Helper function to find the first bin where rate drops below threshold
        def find_first_dropoff(rate_array, bin_edges, threshold):
            # Find all indices where the rate is at or above threshold
            above_indices = np.flatnonzero((rate_array >= threshold) & ~np.isnan(rate_array))

            if len(above_indices) == 0:
                return None  # Never reached the threshold

            last_above_idx = above_indices[-1]

            # Return the first point below after the peak (last_above_idx + 1)
            dropoff_idx = min(last_above_idx + 1, len(bin_edges) - 1)
            return bin_edges[dropoff_idx]

        # Calculate drop-offs to return them for the broader script
        drop_vel_ct = find_first_dropoff(rate_ct, common_bins, threshold_frac)
        drop_vel_tp = find_first_dropoff(rate_tp, common_bins, threshold_frac)
        print(f"{threshold_pct}% cutoff: {drop_vel_tp} and {drop_vel_ct}")

    # --- Single Axes Plot Setup ---
    fig, ax = plt.subplots(figsize=(12, 6))
    bin_edges = common_bins

    # --- Plotting Logic ---
    if plot_recall:
        # Plot Recall Curves Directly (No Ground Truth)
        ax.step(bin_edges, np.append(rate_ct, rate_ct[-1])*100, label=data1_label, color='blue', linewidth=2, where='post')
        ax.step(bin_edges, np.append(rate_tp, rate_tp[-1])*100, label=data2_label, color='red', linewidth=2, where='post')
        ax.set_ylabel("Recall (%)")
    else:
        # Plot Standard Density Histograms
        norm_counts_gt = counts_gt / counts_gt.sum() if normalize else counts_gt
        norm_counts_ct = counts_ct / counts_ct.sum() if normalize else counts_ct
        norm_counts_tp = counts_tp / counts_tp.sum() if normalize else counts_tp

        ax.step(bin_edges, np.append(norm_counts_gt, norm_counts_gt[-1]), label='Ground truth', color='green',
                linewidth=2, where='post')
        ax.step(bin_edges, np.append(norm_counts_ct, norm_counts_ct[-1]), label=data1_label, color='blue', linewidth=2,
                where='post')
        ax.step(bin_edges, np.append(norm_counts_tp, norm_counts_tp[-1]), label=data2_label, color='red', linewidth=2,
                where='post')

        label = ""
        if log:
            label = "Log "
            plt.yscale('log')
        label += "Density" if normalize else "Detections"
        ax.set_ylabel(label)

    # --- General Plot Styling ---
    ax.set_xlabel(xlabel)
    ax.legend(loc='upper right')

    plt.tight_layout()

    # --- Conditional Axis Limits ---
    if not plot_recall:
        plt.xlim((0, max_velocity))
    if plot_recall:
        plt.xlim(0,max_velocity-10)
    ax.set_ylim(0)

    plt.savefig(filename)
    plt.show()

    return drop_vel_ct, drop_vel_tp

def load_ctrack_files(files, crop_transform = np.zeros(3)):
    """Load per-frame CTrack reconstruction `.npy` files into positions, velocities, radii, and attenuations.

    Each file is a pickled dict with key 'reconstruction' = (control_points[N,2,3], (radii[N], attenuations[N])).
    Per particle, the two control points are reduced to one position (their mean, optionally shifted by
    `crop_transform`) and one velocity (endpoint difference: end - start).

    Returns:
        ctrack_pos: list of per-frame position arrays.
        ctrack_vel: list of per-frame velocity arrays.
        ctrack_pos_all: all positions concatenated across frames.
        ctrack_vel_all: all velocities concatenated across frames.
        ctrack_vel_all_magnitude: magnitude of `ctrack_vel_all`.
        (ctrack_radius, ctrack_att): lists of per-frame radius and attenuation arrays.
    """
    ctrack_pos, ctrack_vel = [], []
    ctrack_radius, ctrack_att = [], []
    for file in files:
        ctrack_results = np.load(file, allow_pickle=True).item()
        ctrack_recon = ctrack_results['reconstruction'][0]  # Y,2,3
        ctrack_recon -= crop_transform
        ctrack_pos.append(ctrack_recon.mean(axis=1))
        ctrack_vel.append(ctrack_recon[:, -1, :] - ctrack_recon[:, 0, :])

        rad, att = ctrack_results['reconstruction'][1]
        ctrack_radius.append(rad)
        ctrack_att.append(att)
    ctrack_pos_all = np.concatenate(ctrack_pos, axis=0)
    ctrack_vel_all = np.concatenate(ctrack_vel, axis=0)

    ctrack_vel_all_magnitude = np.linalg.norm(ctrack_vel_all, axis=1)
    return ctrack_pos, ctrack_vel, ctrack_pos_all, ctrack_vel_all, ctrack_vel_all_magnitude, (ctrack_radius, ctrack_att)

def _binned_mean_step_line(ax, x, y, num_bins=50, bin_edges=None, xlabel="", ylabel="", c='b', label=""):
    """Bin `x` (dropping non-finite `x`/`y` pairs) and draw a step line of the mean `y` per bin onto `ax`.

    Shared by `plot_abs_velocity_errors` and `plot_angle_err`, which each plot a binned-mean error curve
    onto a caller-supplied axis.
    Returns the bin edges used, so a second call can reuse them for a shared binning.
    """
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    # Handle case where all data might be filtered out
    if len(x) == 0:
        return

    bins = bin_edges if bin_edges is not None else num_bins
    hist, bin_edges = np.histogram(x, bins=bins)
    bin_indices = np.digitize(x, bin_edges[:-1])

    bin_centers = []
    mean_y = []

    for i in range(1, num_bins + 1):
        y_in_bin = y[bin_indices == i]
        if len(y_in_bin) > 0:
            mean_val = np.mean(y_in_bin)
            center = (bin_edges[i - 1] + bin_edges[i]) / 2
            bin_centers.append(center)
            mean_y.append(mean_val)

    ax.step(bin_centers, mean_y, linewidth=2, where='post', color=c, label=label)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return bin_edges

def plot_abs_velocity_errors(ctracks_package, trackpy_package, save_file="", plot_trackpy=False, ax=None):
    """Plot mean absolute velocity error vs. ground-truth velocity magnitude, with relative-error reference lines.

    `ctracks_package`/`trackpy_package` are (ground_truth_velocity, reconstructed_velocity) pairs. If
    `plot_trackpy` is False, only the CTracks error curve is plotted and TrackPy's max detected velocity
    is drawn as a reference line instead.
    """
    vel_error_ct = np.abs(ctracks_package[1] - ctracks_package[0])

    # Set up plot
    ax_provided = ax is not None
    xlabel = None
    label1 = None
    label2 = None

    if not ax_provided:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.rcParams.update({'font.size': 18})
        xlabel = "Velocity magnitude (vox/frame)"
        label1 = "CTracks"
        label2 = "RDL"

    ### 1. Plot CTracks error
    bin_edges = _binned_mean_step_line(ax, ctracks_package[0], vel_error_ct,
                        xlabel=xlabel,
                        ylabel="Velocity error (vox/frame)",
                        c='blue', label=label1)

    ### 2. Conditionally plot Trackpy error
    if plot_trackpy:
        # Assuming trackpy_package structure is identical to ctracks_package
        vel_error_tp = np.abs(trackpy_package[1] - trackpy_package[0])
        _binned_mean_step_line(ax, trackpy_package[0], vel_error_tp,
                            c='red', label=label2, bin_edges = bin_edges)

    ### 3. Plot vertical line and text at max velocity of Trackpy
    if not plot_trackpy:
        tp_velocities = trackpy_package[0][np.isfinite(trackpy_package[0])]
        if len(tp_velocities) > 0:
            max_tp_vel = np.max(tp_velocities)
            # No label here so it stays out of the legend
            ax.axvline(x=max_tp_vel, color='red', linestyle='--', linewidth=2)

            ax.text(max_tp_vel, 0.95, ' Classical\nmaximum', color='red',
                    transform=ax.get_xaxis_transform(), va='top', ha='left', fontsize=14)

    ### 4. Smart-clip the fan lines and overlap text
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    xmax = ax.get_xlim()[1]
    ymax = ax.get_ylim()[1]

    if plot_trackpy:
        rel_errs_pct = [2.5, 5.0, 7.5, 10, 12.5,15,20]
    else:
        rel_errs_pct = [2.5, 5.0, 7.5]

    for i, pct in enumerate(rel_errs_pct):
        slope = pct / 100.0
        y_at_xmax = slope * xmax

        # Determine where the line hits the boundary
        if y_at_xmax <= ymax:
            x_end = xmax
            y_end = y_at_xmax
        else:
            x_end = ymax / slope
            y_end = ymax

        # Only assign the label to the very first line (i == 0)
        line_label = 'Relative error' if i == 0 else None

        # Draw the line exactly to the edge intersection
        ax.plot([0, x_end], [0, y_end], 'k:', alpha=0.6, label=line_label)

        # Position text 92% of the way along the line
        text_x = x_end * 0.92
        text_y = text_x * slope

        ax.text(text_x, text_y, f"{pct}%", color='black', alpha=0.9,
                ha='center', va='center', fontsize=14,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

    # Turn the legend back on
    ax.legend(loc='best', framealpha=0.9, frameon=True)


    if save_file and not ax_provided:
        plt.tight_layout()
        plt.savefig(save_file)
        plt.show()

def plot_angle_err(ctracks_package, trackpy_package, savefile="", normalise=False, detect_range=1.73,
                   plot_trackpy=False, ax = None):
    """Plot mean velocity-direction error vs. magnitude for CTracks (and optionally TrackPy).

    `ctracks_package`/`trackpy_package` are (ground_truth_velocity, reconstructed_velocity) pairs.
    If `normalise` is True, angle error is expressed as a percentage of the theoretical maximum
    angle error achievable at `detect_range` detection tolerance, and flat reference lines are
    drawn; otherwise absolute-degree arcsine reference curves are drawn.
    """
    # 1. Inner function to calculate magnitude and conditionally normalize angle error
    def calc_err_angle(gt_vel, recon_vel):
        gt_vel_mag = np.linalg.norm(gt_vel, axis=1)
        recon_vel_mag = np.linalg.norm(recon_vel, axis=1)
        ctracks_dot = np.sum(gt_vel * recon_vel, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            cos_theta = np.clip(ctracks_dot / (gt_vel_mag * recon_vel_mag), -1.0, 1.0)
            angle_error = np.rad2deg(np.arccos(cos_theta))

            # If normalise is true, convert the raw degrees to a % of the theoretical max
            if normalise:
                ratio = np.clip(2 * detect_range / gt_vel_mag, -1.0, 1.0)
                max_angle = np.rad2deg(np.arcsin(ratio))
                angle_error = (angle_error / max_angle) * 100

        return gt_vel_mag, angle_error

    # 2. Extract data (Changed '_' to 'err_angle_tp' to save the Trackpy error data)
    gt_mag_ct, err_angle_ct = calc_err_angle(*ctracks_package)
    gt_mag_tp, err_angle_tp = calc_err_angle(*trackpy_package)

    # Set up plot
    ax_provided = ax is not None

    if not ax_provided:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.rcParams.update({'font.size': 18})

    # 3. Plot CTracks error as a step histogram (shared binned-mean helper, see `_binned_mean_step_line`)
    ylabel_text = "Normalized Angle Error (%)" if normalise else "Mean Velocity error angle (°)"
    bin_edges = _binned_mean_step_line(ax, gt_mag_ct, err_angle_ct,
                        xlabel="Velocity magnitude (vox/frame)",
                        ylabel=ylabel_text,
                        c='blue', label="CTracks")

    # NEW: Conditionally plot Trackpy angle error
    if plot_trackpy:
        _binned_mean_step_line(ax, gt_mag_tp, err_angle_tp,
                            c='red', label="RDL", bin_edges = bin_edges)

    # 5. Plot vertical line for Trackpy's maximum velocity
    if not plot_trackpy:
        finite_tp_mags = gt_mag_tp[np.isfinite(gt_mag_tp)]
        if len(finite_tp_mags) > 0:
            max_tp_vel = np.max(finite_tp_mags)
            ax.axvline(x=max_tp_vel, color='red', linestyle='--', linewidth=2)
            ax.text(max_tp_vel, 0.95, ' Classical maximum', color='red',
                    transform=ax.get_xaxis_transform(), va='top', ha='left', fontsize=14)

    # Smart-clip limits
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    xmax = ax.get_xlim()[1]



    ymax = ax.get_ylim()[1]

    # 6. Generate the Reference Lines (Flat if normalized, Curved if absolute)
    angle_pcts = [25, 50, 100]

    if not normalise:
        # Draw the Arcsine curves for absolute degrees
        x_curve = np.linspace(0.01, xmax, 500)
        ratio = np.clip(2 * detect_range / x_curve, -1.0, 1.0)
        max_angle_curve = np.rad2deg(np.arcsin(ratio))

        for i, pct in enumerate(angle_pcts):
            y_curve = (pct / 100.0) * max_angle_curve
            valid_idx = y_curve <= (ymax * 1.05)
            line_label = 'Relative max angle' if i == 0 else None

            ax.plot(x_curve[valid_idx], y_curve[valid_idx], 'k:', alpha=0.6, label=line_label)

            text_x = xmax * 0.92
            eval_ratio = np.clip(2 * detect_range / text_x, -1.0, 1.0)
            text_y = (pct / 100.0) * np.rad2deg(np.arcsin(eval_ratio))

            if text_y <= ymax:
                ax.text(text_x, text_y, f"{pct}%", color='black', alpha=0.9,
                        ha='center', va='center', fontsize=14,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

    # Uncommented the legend so 'Trackpy' and 'CTracks' both appear
    if plot_trackpy and not ax_provided:
        ax.legend(loc='lower right', framealpha=0.7, frameon=True)


    if savefile and not ax_provided:
        plt.tight_layout()
        plt.savefig(savefile)
        plt.show()

def plot_velocity_err_dist(ctracks_package, trackpy_package=None, use_percentage=True, plot_trackpy=True, savefile = ""):
    """Plot velocity-error distribution histograms (absolute or percentage) with 90% CI markers.

    `ctracks_package`/`trackpy_package` are (ground_truth_velocity, reconstructed_velocity) pairs.
    """
    plt.figure(figsize=(10, 6))

    # 1. Dynamically build our datasets list
    # CTracks is always included
    datasets = [('CTracks', ctracks_package, 'blue')]

    # Add Trackpy only if the toggle is True AND the data was provided
    if plot_trackpy and trackpy_package is not None:
        datasets.append(('Trackpy', trackpy_package, 'red'))

    # 2. Pre-process data to find global bounds
    processed_data = {}
    all_valid_errors = []

    for name, package, color in datasets:
        # Calculate raw absolute error
        err = package[1] - package[0]

        # Convert to percentage if toggled
        if use_percentage:
            with np.errstate(divide='ignore', invalid='ignore'):
                err = (err / package[0]) * 100

        # Filter for finite values
        valid_err = err[np.isfinite(err)]
        processed_data[name] = valid_err
        all_valid_errors.extend(valid_err)

    # 3. Determine plot bounds based on whatever data is in the list
    if use_percentage:
        min_val, max_val = -100, 100
    else:
        # Calculate absolute min and max dynamically
        all_valid_errors = np.array(all_valid_errors)
        # Safety check in case of empty arrays
        if len(all_valid_errors) > 0:
            min_val = np.min(all_valid_errors)
            max_val = np.max(all_valid_errors)
        else:
            min_val, max_val = -10, 10

            # 4. Plotting loop
    for name, package, color in datasets:
        valid_err = processed_data[name]
        bounded_err = valid_err[(valid_err >= min_val) & (valid_err <= max_val)]

        # Safety check to ensure we have data in the bounds before calculating percentiles
        if len(bounded_err) > 0:
            ci_lower, ci_upper = np.percentile(bounded_err, [5, 95])

            plt.hist(bounded_err, bins=50, histtype='step', linewidth=1.5, color=color,
                     label=f'{name}', density=True)

            if not plot_trackpy:
                axvcolor = 'k'
            else: axvcolor = color
            plt.axvline(x=ci_lower, color=axvcolor, linestyle=':', linewidth=2)
            plt.axvline(x=ci_upper, color=axvcolor, linestyle=':', linewidth=2)

    # 5. Universal formatting
    plt.axvline(x=0, color='black', linestyle='-', linewidth=2)
    plt.plot([], [], color='gray', linestyle=':', linewidth=2, label='90% CI')

    ax = plt.gca()
    if use_percentage:
        plt.xlabel("Error (%)")
        ax.xaxis.set_major_locator(MultipleLocator(20))
    else:
        plt.xlabel("Absolute Error")

    plt.ylabel("Probability density")


    plt.legend(loc='upper right')
    plt.savefig(savefile)
    plt.show()


def calculate_and_print_metrics_comparison(
        sim_field,
        test_field_1,
        test_field_2,
        pore_mask=None,
        method_1_name="RDL",
        method_2_name="CTracks"
):
    """
    Calculates and prints comparison metrics for two alternative methods against a simulation baseline,
    including the relative improvement from Method 1 to Method 2.
    """
    # Apply the mask if provided, otherwise flatten the entire domain
    if pore_mask is not None:
        fluid_mask = pore_mask.astype(bool)
        sim_flat = sim_field[fluid_mask]
        test1_flat = test_field_1[fluid_mask]
        test2_flat = test_field_2[fluid_mask]
        status = "Fluid Domain Only (Using Provided Pore Mask)"
    else:
        sim_flat = sim_field.flatten()
        test1_flat = test_field_1.flatten()
        test2_flat = test_field_2.flatten()
        status = "Full Domain (No Mask Applied)"

    # Avoid empty arrays
    if len(sim_flat) == 0:
        print("Error: Mask contains no fluid nodes (all False).")
        return None

    def compute_metrics(test_flat):
        mae = mean_absolute_error(sim_flat, test_flat)
        mse = mean_squared_error(sim_flat, test_flat)
        rmse = np.sqrt(mse)
        corr, _ = pearsonr(sim_flat, test_flat)
        r2 = r2_score(sim_flat, test_flat)

        with np.errstate(divide='ignore', invalid='ignore'):
            # Define a tiny threshold to ignore near-zero simulation velocities (e.g., 1e-5)
            # Adjust this threshold based on your typical flow field velocity scales
            valid_sim_nodes = np.abs(sim_flat) > 1e-5

            if np.any(valid_sim_nodes):
                relative_error = np.abs(sim_flat[valid_sim_nodes] - test_flat[valid_sim_nodes]) / sim_flat[
                    valid_sim_nodes]
                # Use np.nanmean to safely handle any rogue NaNs and get a single scalar
                mre = float(np.nanmean(relative_error[np.isfinite(relative_error)]))
            else:
                mre = 0.0

        return {"MAE": mae, "RMSE": rmse, "Pearson_R": corr, "R2": r2, "MRE": mre}

    # Compute metrics for both methods
    m1 = compute_metrics(test1_flat)
    m2 = compute_metrics(test2_flat)

    # Calculate Relative Improvement (%) from Method 1 to Method 2
    # For errors (MAE, RMSE, MRE), lower is better -> (M1 - M2) / M1
    # For fit metrics (Pearson R, R2), higher is better -> (M2 - M1) / M1
    # Note: If M1 is 0 or R2 is negative, percentage improvement can get wonky, handled implicitly or left as standard.
    improvement = {}
    for metric in ["MAE", "RMSE", "MRE"]:
        improvement[metric] = ((m1[metric] - m2[metric]) / m1[metric]) * 100 if m1[metric] != 0 else 0.0

    for metric in ["Pearson_R", "R2"]:
        # Standard relative change for fitness metrics
        improvement[metric] = ((m2[metric] - m1[metric]) / abs(m1[metric])) * 100 if m1[metric] != 0 else 0.0

    # Print Dashboard
    print(f"\n==========================================================================")
    print(f" Performance Metrics Comparison: {method_1_name} vs {method_2_name}")
    print(f" Status: {status}")
    print(f" Evaluated Nodes: {len(sim_flat):,} / {sim_field.size:,}")
    print(f"==========================================================================")
    print(f"{'Metric':<28} | {method_1_name:<12} | {method_2_name:<12} | {'Improvement':<12}")
    print(f"--------------------------------------------------------------------------")
    print(f"{'Mean Absolute Error (MAE)':<28} | {m1['MAE']:<12.5f} | {m2['MAE']:<12.5f} | {improvement['MAE']:+11.2f}%")
    print(
        f"{'Root Mean Sq. Error (RMSE)':<28} | {m1['RMSE']:<12.5f} | {m2['RMSE']:<12.5f} | {improvement['RMSE']:+11.2f}%")
    print(
        f"{'Pearson Correlation (R)':<28} | {m1['Pearson_R']:<12.5f} | {m2['Pearson_R']:<12.5f} | {improvement['Pearson_R']:+11.2f}%")
    print(f"{'R-squared (R²)':<28} | {m1['R2']:<12.5f} | {m2['R2']:<12.5f} | {improvement['R2']:+11.2f}%")
    print(f"{'Mean Relative Error (MRE)':<28} | {m1['MRE']:<12.5f} | {m2['MRE']:<12.5f} | {improvement['MRE']:+11.2f}%")
    print(f"==========================================================================\n")

    return {"Method_1": m1, "Method_2": m2, "Improvement_Percent": improvement}
