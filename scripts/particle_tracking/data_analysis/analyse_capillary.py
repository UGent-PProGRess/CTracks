"""Analyse capillary particle tracks: fit Poiseuille flow profiles from CTracks/TrackPy detections,
compare them against the theoretical prediction, and plot flow-field comparisons.
"""
import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ctrex.utils.filetools import filelist
from ctracks.postprocessing import link_and_filter_functions as tpf

from scripts.particle_tracking.data_analysis.analysis_functions import load_ctrack_files, plot_two_velocity_slices, plot_velocity_distributions, \
    calc_mean_distances, default_velocity_cmap


# ============================================================================
# User-adjustable configuration
# ============================================================================

# --- Run options -------------------------------------------------------------
# Neither flag is currently read by this script's flow; kept as placeholders for parity
# with the other analyse_*.py scripts.
RUN_OPTIONS = {
    "fit_trackpy": False,
    "calc_interp": False,
}

# File type for this script's own figures - "png" or "svg".
FIGURE_EXT = "png"

# --- Sample geometry (capillary) ----------------------------------------------
SAMPLE_GEOMETRY = {
    "roi_height": slice(0, 800),
    "roi_width": slice(100, 250),
    "mask_threshold": 0.5,
}
SAMPLE_GEOMETRY["roi"] = (SAMPLE_GEOMETRY["roi_height"], SAMPLE_GEOMETRY["roi_width"])
SAMPLE_GEOMETRY["crop_transform"] = np.array(
    [SAMPLE_GEOMETRY["roi_width"].start, SAMPLE_GEOMETRY["roi_width"].start, SAMPLE_GEOMETRY["roi_height"].start]
) * -1
SAMPLE_GEOMETRY["trackpy_crop_transform"] = SAMPLE_GEOMETRY["crop_transform"]

# --- Acquisition / calibration parameters --------------------------------------
ACQUISITION_PARAMS = {
    "voxel_size_um": 12.2,  # micrometers per voxel
    "voxel_size_m": 12.2e-6,  # meters per voxel
    "frame_time_s": 30.0,  # seconds per frame/scan
}

# --- Poiseuille fit / calibration constants ------------------------------------
R_KNOWN = 46.74625  # capillary radius (voxels), 46.75 +/- 1.78 (SE, Student's t-test)
NUM_RADIAL_BINS = 20
REL_CI_D = 1.7798 / 93.4925  # relative 95% CI on the capillary diameter measurement
REL_CI_Q = 0.005  # relative 95% CI on the syringe-pump flow rate

# --- Data root - change this if CTracksSubmissionData is moved or mounted under a
# different drive/path (e.g. moving to a new PC). ---------------------------------
DATA_ROOT = "J:\\"

# --- Base directory  --------------------------------------------------
directory = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Capillary")

# --- Where this script's own figures/plots are saved -----------------------------
ANALYSIS_RESULTS_DIR = os.path.join(directory, "analysis_results")
os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)

MASK_FILE = os.path.join(directory, 'segmentation.tif')

# --- Per-flow-rate datasets ------------------------------------------------------
DATASETS = {
    "60nlmin": dict(
        ctracks_frame_range=range(0, 40),
        basefolder=os.path.join(directory, "60nlmin"),
        ctracks_recon_dir=os.path.join(directory, "60nlmin", "ctracks_trajectories"),
        trackpy_recon_file=os.path.join(directory, "60nlmin", "rdl_results", "capillary_60nlmin_velocityPoints"),
        vel_maxima=(5, 5, 10),
        interp_vrange=(1, 5),
        flow_rate=60,  # nl/min
    ),
    "180nlmin": dict(
        ctracks_frame_range=range(0, 20),
        basefolder=os.path.join(directory, "180nlmin"),
        ctracks_recon_dir=os.path.join(directory, "180nlmin", "ctracks_trajectories"),
        trackpy_recon_file=os.path.join(directory, "180nlmin", "rdl_results", "capillary_180nlmin_velocityPoints"),
        vel_maxima=(5, 5, 24),
        interp_vrange=(5, 15),
        flow_rate=180,  # nl/min
    ),
    "1ulmin": dict(
        ctracks_frame_range=range(0, 20),
        basefolder=os.path.join(directory, "1ulmin"),
        ctracks_recon_dir=os.path.join(directory, "1ulmin", "ctracks_trajectories"),
        trackpy_recon_file=os.path.join(directory, "1ulmin", "rdl_results", "capillary_1ulmin_velocityPoints"),
        vel_maxima=(5, 5, 120),
        interp_vrange=(20, 100),
        flow_rate=1000,  # nl/min
    ),
}
# All three flow-rate datasets are analysed together (for the multi-panel comparison plots),
# in this display order:
DATASET_ORDER = ["60nlmin", "180nlmin", "1ulmin"]
datasets = [DATASETS[name] for name in DATASET_ORDER]
NAMES = ["60 nl/min", "180 nl/min", "1 \u03bcl/min"]
PANEL_LABELS = ['(a)', '(b)', '(c)']

# --- CTracks filter/link parameters (shared across datasets; velocity_maxima is per-dataset) ---
FILTER_PARAMS = dict(
    min_radius=20, min_atten=30, velocity_clip=10, k_neighbors=5, uod_threshold=2, eps=0.5,
    threshold_reverse_vel=None, isolation_search_radius=15, isolation_min_neighbors=3,
    isolation_per_frame=False,
)

# --- Flow-field slice plotting styling ------------------------------------------
MASK_ZOOM_FACTOR = 0.5  # downsampling factor applied to the mask for the interpolated-field plots
FIELD_PLOT_X_CROP = 15
FIELD_PLOT_Y_CROP = 30

plt.rcParams['font.size'] = 18


# ============================================================================
# Implementation
# ============================================================================

def find_center_line_mask(mask):
    """Find the (x, y) centerline of a capillary mask, one point per z-slice.

    For each z-slice, the centerline point is the centroid of the True voxels in that slice.

    Returns:
        (Z, 2) array of (x, y) midpoints, one row per z-slice.
    """
    Z_slices = mask.shape[0]
    midpoint_array = np.zeros((Z_slices, 2))

    for z in range(Z_slices):
        slice_2d = mask[z]
        y_x_coords = np.argwhere(slice_2d)
        mid_y, mid_x = np.mean(y_x_coords, axis=0)
        midpoint_array[z] = np.array([mid_x, mid_y])
    print(f"Generated {Z_slices} midpoints, one for each z-slice.")
    return midpoint_array

def find_radial_distance(coords, midpoint_array):
    """Compute each point's radial distance (in x-y) from the capillary centerline at its z-slice."""
    z_indices = np.floor(coords[:, 2]).astype(int)
    Z_max = midpoint_array.shape[0] - 1
    z_indices = np.clip(z_indices, 0, Z_max)
    corresponding_centers = midpoint_array[z_indices]
    diff_vector = coords[:,0:2] - corresponding_centers
    radial_distances_sq = np.sum(diff_vector[:, 0:2] ** 2, axis=1)
    radial_distances = np.sqrt(radial_distances_sq)

    return radial_distances


def generate_theoretical_poiseuille(pore_mask,max_radius,coefficient) :
    """
    Generates a 3D theoretical Poiseuille flow field by calling the user's
    find_center_line_mask function internally.
    """
    nz, ny, nx = pore_mask.shape

    # 1. Call your function directly
    midpoint_array = find_center_line_mask(pore_mask)

    # 2. Unpack your midpoint array (column 0 is X, column 1 is Y)
    mid_x_1d = midpoint_array[:, 0]
    mid_y_1d = midpoint_array[:, 1]

    # 3. Reshape for fast 3D broadcasting
    mid_x_3d = mid_x_1d[:, np.newaxis, np.newaxis]
    mid_y_3d = mid_y_1d[:, np.newaxis, np.newaxis]

    # 4. Generate spatial coordinates
    _, y_coords, x_coords = np.indices((nz, ny, nx))

    # 5. Calculate radial distance at every voxel
    radial_distances = np.sqrt((y_coords - mid_y_3d) ** 2 + (x_coords - mid_x_3d) ** 2)

    # 6. Apply the theoretical Poiseuille equation: v(r) = C * (R^2 - r^2)
    velocity_magnitude = coefficient * ((max_radius ** 2) - (radial_distances ** 2))

    # 7. Clean up boundaries and enforce the mask
    velocity_magnitude = np.clip(velocity_magnitude, a_min=0, a_max=None)
    final_magnitude_volume = velocity_magnitude * pore_mask

    return final_magnitude_volume

def fit_poiseuille_umax_only(r_data, u_data, R_known=47.0):
    """Fit a Poiseuille profile v(r) = u_max * (1 - (r/R_known)^2) with only u_max free.

    Not currently called by this script's main flow; kept as a simpler alternative to the
    weighted C-coefficient fit used in `plot_velocity_radial_distribution`.
    """
    def poiseuille_velocity_fixed_R(r, u_max):
        return u_max * (1 - (r / R_known) ** 2)

    u_max_guess = np.max(u_data)

    popt, pcov = curve_fit(
        f=poiseuille_velocity_fixed_R,
        xdata=r_data,
        ydata=u_data,
        p0=[u_max_guess]
    )

    u_max_fit = popt[0]

    sim_xdata = np.linspace(min(r_data), max(r_data), 100)
    sim_ydata = poiseuille_velocity_fixed_R(sim_xdata, u_max_fit)

    factor_A = u_max_fit / (R_known ** 2)

    print(f"Fit u_max parameter: {u_max_fit:.4f}")

    legend = "Fit ΔP/4μL = " + str(np.round(factor_A, decimals=4))

    return sim_xdata, sim_ydata, legend


def plot_velocity_radial_distribution(ax, ctracks_data, trackpy_data, num_bins=20,
                                      R_known=47.0, flow_rate=None, show_theory_ci=True, trim_radius=40.0):
    """
    Plots velocity radial profiles with relative error in the legend and
    prints the raw absolute values directly to the console.
    """
    # Unpack tracking data
    _, _, _, _, ctracks_vel_mag, ctracks_rad_dist = ctracks_data
    _, _, _, _, trackpy_vel_mag, trackpy_rad_dist = trackpy_data

    all_rad_dist = np.concatenate([ctracks_rad_dist, trackpy_rad_dist])
    bins = np.linspace(all_rad_dist.min(), all_rad_dist.max(), num_bins + 1)

    def process_and_bin_weighted(radial_dist, velocity_mag, bins):
        bin_indices = np.digitize(radial_dist, bins)
        binned_distances, binned_velocities, bin_counts = [], [], []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.any(mask):
                binned_distances.append((bins[i - 1] + bins[i]) / 2)
                binned_velocities.append(np.mean(velocity_mag[mask]))
                bin_counts.append(np.sum(mask))
        return np.array(binned_distances), np.array(binned_velocities), np.array(bin_counts)

    dist_c, vel_c, counts_c = process_and_bin_weighted(ctracks_rad_dist, ctracks_vel_mag, bins)
    dist_t, vel_t, counts_t = process_and_bin_weighted(trackpy_rad_dist, trackpy_vel_mag, bins)

    # Base profile function: v(r) = C * (R^2 - r^2)
    def poiseuille_profile(r, C):
        return C * (R_known ** 2 - r ** 2)

    fit_x = np.linspace(0, R_known, 100)

    def fmt_2_sigfigs(val):
        """Format a percentage value to 2 significant digits."""
        if val == 0:
            return "0.0%"
        first_sig_digit = int(np.floor(np.log10(abs(val))))
        decimals_needed = max(0, 1 - first_sig_digit)
        return f"{val:.{decimals_needed}f}%"

    # --- 1. THEORETICAL EXPECTATION ---
    C_pixel_theory = 1.0
    if flow_rate is not None:
        D_voxels = 2.0 * R_known

        D_m = D_voxels * ACQUISITION_PARAMS['voxel_size_m']
        Q_m3_s = (flow_rate * 1e-12) / 60.0
        C_si = (32 * Q_m3_s) / (np.pi * (D_m ** 4))
        C_pixel_theory = C_si * ACQUISITION_PARAMS['frame_time_s'] * ACQUISITION_PARAMS['voxel_size_m']

        rel_ci_C = np.sqrt(REL_CI_Q ** 2 + (4 * REL_CI_D) ** 2)
        ci_95_theory = rel_ci_C * C_pixel_theory

        ax.plot(fit_x, poiseuille_profile(fit_x, C_pixel_theory), linewidth=1.5,
                color='black', linestyle='--', alpha = 0.7)

        if show_theory_ci:
            ax.fill_between(fit_x,
                            poiseuille_profile(fit_x, C_pixel_theory - ci_95_theory),
                            poiseuille_profile(fit_x, C_pixel_theory + ci_95_theory),
                            color='gray', alpha=0.15)

    # --- 2. WEIGHTED CTRACKS FIT ---
    mask_trim_c = dist_c <= trim_radius
    sigma_weights_c = 1.0 / np.sqrt(counts_c[mask_trim_c])
    popt_c, _ = curve_fit(poiseuille_profile, dist_c[mask_trim_c], vel_c[mask_trim_c],
                          p0=[0.001], sigma=sigma_weights_c, absolute_sigma=False)
    C_pixel_ctracks = popt_c[0]

    err_ctracks = ((C_pixel_ctracks - C_pixel_theory) / C_pixel_theory) * 100.0
    ctracks_label = fmt_2_sigfigs(err_ctracks)
    if err_ctracks > 0: ctracks_label = "+" + ctracks_label

    ax.scatter(dist_c, vel_c,marker='o',  s=80,  color='blue',alpha=1.0,label='CTracks',edgecolors='black', zorder= 10, linewidth = 1  )
    ax.plot(fit_x, poiseuille_profile(fit_x, C_pixel_ctracks), linewidth=1.5,
            color='blue', label=ctracks_label, alpha=0.6, linestyle = "--")

    # --- 3. WEIGHTED TRACKPY (RDL) FIT ---
    mask_trim_t = dist_t <= trim_radius
    sigma_weights_t = 1.0 / np.sqrt(counts_t[mask_trim_t])
    popt_t, _ = curve_fit(poiseuille_profile, dist_t[mask_trim_t], vel_t[mask_trim_t],
                          p0=[0.001], sigma=sigma_weights_t, absolute_sigma=False)
    C_pixel_trackpy = popt_t[0]

    err_trackpy = ((C_pixel_trackpy - C_pixel_theory) / C_pixel_theory) * 100.0
    rdl_label = fmt_2_sigfigs(err_trackpy)
    if err_trackpy > 0: rdl_label = "+" + rdl_label

    ax.scatter(dist_t, vel_t, marker='^', s=80, color='red', alpha=1.0, label='RDL', edgecolors='black', zorder= 8, linewidth = 1)

    ax.plot(fit_x, poiseuille_profile(fit_x, C_pixel_trackpy), linewidth=1.5,
            color='red', label=rdl_label, linestyle="--", alpha = 0.6)

    # --- 4. CONSOLE OUTPUT PRINT ---
    print("\n" + "=" * 35)
    print("      ABSOLUTE VALUE METRICS     ")
    print("=" * 35)
    print(f"Theoretical Constant : {C_pixel_theory:.6e}")
    print(f"CTracks Constant     : {C_pixel_ctracks:.6e} ({ctracks_label})")
    print(f"RDL/Trackpy Constant : {C_pixel_trackpy:.6e} ({rdl_label})")
    print("=" * 35 + "\n")

    # Subplot configuration adjustments (No text labels added)
    ax.set_xlim(left=0, right=R_known)
    ax.set_ylim(bottom=0)
    return C_pixel_theory


def vx_to_um(x):
    """Convert a voxel-length value to micrometers."""
    return x * ACQUISITION_PARAMS['voxel_size_um']

def um_to_vx(x):
    """Convert a micrometer-length value to voxels."""
    return x / ACQUISITION_PARAMS['voxel_size_um']

def vxpf_to_ums(y):
    """Convert a voxel/frame velocity to micrometers/second."""
    return (y * ACQUISITION_PARAMS['voxel_size_um']) / ACQUISITION_PARAMS['frame_time_s']

def ums_to_vxpf(y):
    """Convert a micrometer/second velocity to voxels/frame."""
    return (y * ACQUISITION_PARAMS['frame_time_s']) / ACQUISITION_PARAMS['voxel_size_um']


def load_capillary_mask_and_midline():
    """Load the capillary pore mask (cropped to the ROI) and compute its z-wise centerline."""
    mask = tifffile.imread(MASK_FILE) > SAMPLE_GEOMETRY['mask_threshold']
    mask = mask[SAMPLE_GEOMETRY['roi_height'], SAMPLE_GEOMETRY['roi_width'], SAMPLE_GEOMETRY['roi_width']]
    midpoints = find_center_line_mask(mask)
    return mask, midpoints


def load_and_filter_datasets(midpoints):
    """Load, filter and link both TrackPy/RDL and CTracks results for each flow-rate dataset.

    Also plots the per-dataset velocity-distribution comparison and prints mean nearest-neighbor
    separation stats.

    Returns:
        (tp_datas, ct_datas): lists (one entry per dataset, in `datasets` order) of
        (pos, vel, pos_all, vel_all, vel_all_magnitude, radial_distance) tuples.
    """
    tp_datas = []
    ct_datas = []

    for name, dataset in zip(DATASET_ORDER, datasets):
        # load trackpy results
        frame_range = dataset['ctracks_frame_range']
        tp_data = pd.read_csv(dataset['trackpy_recon_file'] + ".csv")  # dataframe particle, frame, z,y,x,vz,vy,vx,velMags
        tp_data.drop(index=0, inplace=True)
        tp_loaded = tpf.convert_trackpy_to_frame_lists(tp_data, frame_range=dataset['ctracks_frame_range'])
        trackpy_radial_distances = find_radial_distance(tp_loaded[2], midpoints)
        tp_loaded = tp_loaded + (trackpy_radial_distances,)
        tp_datas.append(tp_loaded)

        print(f"Loaded {tp_loaded[2].shape[0] / len(frame_range)} average trackpy tracks per frame")

        # load ctrack results
        ctracks_files = filelist(dataset['ctracks_recon_dir'], fmt="scan_%s.npy", sort=True)[frame_range[0]:frame_range[-1]]
        ctracks_loaded = load_ctrack_files(ctracks_files)

        dataframe, ctracks_filtered = (
            tpf.filter_vectors_direct(ctracks_loaded[0], ctracks_loaded[1], ctracks_loaded[-1],
                                      velocity_maxima=dataset['vel_maxima'], **FILTER_PARAMS))
        dataframe.to_csv(os.path.join(dataset['ctracks_recon_dir'], "linked_trajectories.csv"), index=False)

        ctracks_filtered[2][:] += SAMPLE_GEOMETRY['crop_transform']
        ctrack_radial_distances = find_radial_distance(ctracks_filtered[2], midpoints)
        ctracks_filtered = ctracks_filtered + (ctrack_radial_distances,)
        ct_datas.append(ctracks_filtered)
        print(f"Loaded {ctracks_filtered[2].shape[0] / len(ctracks_files)} average ctrack tracks per frame")

        plot_velocity_distributions(ctracks_filtered[3], tp_loaded[3],
                                    save_file=os.path.join(ANALYSIS_RESULTS_DIR, f"velocity_distribution_{name}.{FIGURE_EXT}"))
        print("CTracks - mean separation")
        calc_mean_distances(ctracks_filtered[0])
        print("TrackPy - mean separation")
        calc_mean_distances(tp_loaded[0])
        print("\n\n")

    return tp_datas, ct_datas


def plot_radial_velocity_profiles(ct_datas, tp_datas):
    """Fit and plot the measured Poiseuille radial velocity profile for each flow-rate dataset.

    Draws a 3-panel figure (one panel per dataset) comparing CTracks and RDL radial velocity
    profiles against the theoretical Poiseuille prediction, and stores the fitted theoretical
    coefficient back onto each dataset dict (key 'coeff') for later use in the flow-field plots.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for i in range(len(datasets)):
        pc = plot_velocity_radial_distribution(axs[i], ct_datas[i], tp_datas[i], NUM_RADIAL_BINS,
                                          R_known=R_KNOWN, flow_rate = datasets[i]['flow_rate'],
                                          show_theory_ci = False)

        axs[i].text(0.97, 0.94, PANEL_LABELS[i],
                transform=axs[i].transAxes,  # Use axes coordinates, not data coordinates
                ha='right', va='top',  # Align the top-right corner of the text block
                fontsize=18, fontweight='bold',
                # Add a white background box so data lines don't obscure the text
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='white', boxstyle='round,pad=0.4'))
        axs[i].secondary_yaxis('right', functions=(vxpf_to_ums, ums_to_vxpf))
        datasets[i]['coeff'] = pc

    line_poiseuille = Line2D([0], [0], color='black', linestyle="--", lw=2, alpha=0.7)
    line_ctracks_fit = Line2D([0], [0], color='blue', linestyle="--", lw=1.5, alpha=0.6)
    marker_ctracks_data = Line2D([0], [0], color='blue', marker='o', linestyle='None',
                                 markersize=9, markeredgecolor='black', markeredgewidth=1.0)
    line_rdl_fit = Line2D([0], [0], color='red', linestyle="--", lw=1.5, alpha=0.6)
    marker_rdl_data = Line2D([0], [0], color='red', marker='^', linestyle='None',
                             markersize=9, markeredgecolor='black', markeredgewidth=1)
    legend_handles = [
        line_poiseuille,
        (line_rdl_fit, marker_rdl_data),       # Combines Red Dashed Line + Triangles
        (line_ctracks_fit, marker_ctracks_data) # Combines Blue Dashed Line + Dots
    ]

    legend_labels = ['Poiseuille','RDL','CTracks']
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='lower right',
        bbox_to_anchor=(0.35, 0.39)
    )

    axs[0].secondary_xaxis('top', functions=(vx_to_um, um_to_vx))
    fig.supylabel("Velocity Magnitude (voxels/frame)")
    fig.supxlabel("Radial distance (voxels)")
    # Top label: Horizontally centered (0.5), at the very top (0.98)
    fig.text(0.5, 0.98, 'Radial distance (\u00b5m)',
             ha='center', va='top',
             fontsize=plt.rcParams['figure.labelsize'])

    # Right label: Far right (0.98), Vertically centered (0.5), Rotated faces inward
    fig.text(0.98, 0.5, 'Velocity Magnitude (\u00b5m/s)',
             ha='right', va='center', rotation=270,
             fontsize=plt.rcParams['figure.labelsize'])

    # Constrain tight_layout to leave a safety margin on both the top and right side
    plt.tight_layout(rect=[0, 0, 0.94, 0.94])

    plt.savefig(os.path.join(ANALYSIS_RESULTS_DIR, f"capillary_radial_distance.{FIGURE_EXT}"))
    plt.show()


def main():
    """Analyse capillary particle tracks: filter/link, fit Poiseuille profiles, and plot flow-field comparisons."""
    mask, midpoints = load_capillary_mask_and_midline()
    tp_datas, ct_datas = load_and_filter_datasets(midpoints)
    plot_radial_velocity_profiles(ct_datas, tp_datas)


if __name__ == "__main__":
    main()
