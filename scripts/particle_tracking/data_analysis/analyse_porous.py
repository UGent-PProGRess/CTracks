"""Analyse porous-sample particle tracks: filter/link CTracks results, compare them against
TrackPy/RDL and (optionally) a simulated ground truth, and produce the paper's comparison plots.
"""
import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from scripts.particle_tracking.data_analysis.vis2D.plot_slices_tracks import plot_slices
from scripts.particle_tracking.data_analysis.vis3D.plot_3d_vectors import plot_single, plot_video
from ctrex.utils.filetools import filelist

from scripts.particle_tracking.data_analysis.analysis_functions import load_ctrack_files, plot_velocity_distributions, plot_two_velocity_slices, \
    plot_velocity_field_slice, calc_mean_distances, plot_velocity_distributions_general, \
    calculate_and_print_metrics_comparison
from ctracks.postprocessing.link_and_filter_functions import convert_trackpy_to_frame_lists, filter_vectors_direct


# ============================================================================
# User-adjustable configuration
# ============================================================================

# --- Run options -----------------------------------------------------------------
RUN_OPTIONS = {
    "load_dataframe": False,
    "compare_simulation_distribution": True,
    "compare_simulation_field": True,
    "plot_velfields": True,
    "plot_2D_slices": False,
    "plot_3D_streamlines": True,
    "plot_3D_video": False,
}

# File type for this script's own 2D (matplotlib) figures - "png" or "svg".
# Does not affect the 3D pyvista renders (streamlines/video), which are always png.
FIGURE_EXT = "png"

# --- Sample geometry (viscoelastic porous-flow sample) -----------------------------
SAMPLE_GEOMETRY = {
    "frame_crop": ((224, 224), (80, 80), (80, 80)),
    "roi_height": slice(int((948 - 500) // 2), int((948 + 500) // 2)),
    "roi_width": slice(80, 480),
    "mask_threshold": 0.5,
}
SAMPLE_GEOMETRY["roi"] = (SAMPLE_GEOMETRY["roi_height"], SAMPLE_GEOMETRY["roi_width"])
SAMPLE_GEOMETRY["crop_transform"] = np.array(
    [SAMPLE_GEOMETRY["roi_width"].start, SAMPLE_GEOMETRY["roi_width"].start, SAMPLE_GEOMETRY["roi_height"].start]
) * -1

# --- Data root - change this if CTracksSubmissionData is moved or mounted under a
# different drive/path (e.g. moving to a new PC). ---------------------------------
DATA_ROOT = "J:\\"

# --- Dataset locations -------------------------------------------------------------
IMAGE_FOLDER_BASE = "recon_"
BASEDIR = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Porous glass")
SEG_PATH_CROPPED = os.path.join(BASEDIR, "segmentation_cropped.tif")


def analysis_results_dir(cfg):
    """Where this dataset's own figures/plots are saved: an `analysis_results` subfolder
    inside that dataset's own basefolder (not shared across datasets)."""
    result_dir = os.path.join(cfg['basefolder'], "analysis_results")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


# --- Simulated ground-truth comparison ----------------------------------------------
GROUND_TRUTH_TRACKS_FILE = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Simulated\ground_truth_tracks.npy")
SIM_FIELD_FILE = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Simulation\fields\Umag.tif")
ACQUISITION_PARAMS = {
    "sim_substeps_per_frame": 851,  # number of simulated sub-steps between each recon frame
}
SIM_FIELD_LOG_PERCENTILE = 99  # percentile used as the "peak" when log-normalizing velocity fields
DETECTION_VELOCITY_THRESHOLD = 1  # min velocity magnitude counted as a "detection" in the per-frame count print
MASK_ZOOM_FACTOR = 0.5  # downsampling factor applied to masks/fields for the interpolated-field plots
DIFF_FIELD_VRANGE = (0, 1)
DIFF_FIELD_ABS_VRANGE = (0.2, 1)

# --- Per-flow-rate datasets (particle-filter/link params live in each dataset's `filter_params`) ---
DATASETS = {
    "60nlmin": dict(
        recon_folder=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Porous glass\60nlmin\recon"),
        ctracks_frame_range=range(0, 90),
        basefolder=os.path.join(BASEDIR, "60nlmin"),
        ctracks_recon_dir=os.path.join(BASEDIR, "60nlmin", "ctracks_trajectories"),
        mask_file=os.path.join(BASEDIR, "60nlmin", "util_recons", "registered_mask.tif"),
        trackpy_recon_file=os.path.join(BASEDIR, "60nlmin", "rdl_results", "visco_sample2_60nlmin_velocityPoints"),
        interp_vrange=(0.5, 3.5),
        slice_shift=0,
        filter_params={
            "min_radius": 20, "min_atten": 30, "velocity_clip": 10, "velocity_maxima": (7.5, 7.5, 10),
            "threshold_reverse_vel": None,
            "k_neighbors": 5, "uod_threshold": 1.5, "eps": 0.5,
            "isolation_search_radius": 10, "isolation_min_neighbors": 5, "isolation_per_frame": False,
        },
        clim=(1, 4),
        velocity_split=2,
        plot_crop=[0, 500, 0, 500, 200, 500],
    ),
    "180nlmin": dict(
        recon_folder=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Porous glass\180nlmin\recon"),
        ctracks_frame_range=range(0, 60),
        basefolder=os.path.join(BASEDIR, "180nlmin"),
        ctracks_recon_dir=os.path.join(BASEDIR, "180nlmin", "ctracks_trajectories"),
        mask_file=os.path.join(BASEDIR, "180nlmin", "util_recons", "registered_mask.tif"),
        trackpy_recon_file=os.path.join(BASEDIR, "180nlmin", "rdl_results", "visco_sample2_180nlmin_velocityPoints"),
        interp_vrange=(1, 6),
        slice_shift=0,
        filter_params={
            "min_radius": 20, "min_atten": 30, "velocity_clip": 5, "velocity_maxima": (7.5, 7.5, 15),
            "threshold_reverse_vel": None,
            "k_neighbors": 10, "uod_threshold": 3, "eps": 0.5,
            "isolation_search_radius": 30, "isolation_min_neighbors": 1, "isolation_per_frame": True,
        },
        clim=(1, 15),
        velocity_split=4,
        plot_crop=[50, 250, 50, 250, 50, 250],
    ),
    "1ulmin": dict(
        recon_folder=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Porous glass\1ulmin\recon"),
        ctracks_frame_range=range(0, 20),
        basefolder=os.path.join(BASEDIR, "1ulmin"),
        ctracks_recon_dir=os.path.join(BASEDIR, "1ulmin", "ctracks_trajectories"),
        mask_file=os.path.join(BASEDIR, "1ulmin", "util_recons", "registered_mask.tif"),
        trackpy_recon_file=os.path.join(BASEDIR, "1ulmin", "rdl_results", "visco_sample2_1ulmin_velocityPoints"),
        interp_vrange=(5, 15),
        slice_shift=-50,
        filter_params={
            "min_radius": 0, "min_atten": 20, "velocity_clip": 10, "velocity_maxima": (25, 25, 50),
            "threshold_reverse_vel": None,
            "k_neighbors": 5, "uod_threshold": 3, "eps": 0.5,
            "isolation_search_radius": 15, "isolation_min_neighbors": 10, "isolation_per_frame": False,
        },
        clim=(1, 30),
        velocity_split=4,
        plot_crop=[0, 500, 0, 500, 200, 500],
    ),
}
ACTIVE_DATASET = "180nlmin"

# --- 3D streamline/video plot styling (dataset-independent parts) --------------------
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

def plot_shape_distributions(shape_before, linked_dataframe, num_bins = 20):
    """Plot radius and attenuation histograms before vs. after filtering/linking (not currently called)."""
    before_rad, before_att = shape_before #num frames, N each
    before_rad = np.concatenate(before_rad)
    before_att = np.concatenate(before_att)
    after_rad, after_att = linked_dataframe['radius'], linked_dataframe['attenuation']

    #radius
    min_val = min(np.min(before_rad), np.min(after_rad))
    max_val = max(np.max(before_rad), np.max(after_rad))
    bins = np.linspace(min_val, max_val, num_bins)

    plt.figure()
    ax = plt.gca()
    ax.hist(before_rad, bins=bins, alpha=1, color='b', linewidth=2, histtype='step', label='Before linking', density=False)
    ax.axvline(np.mean(before_rad), color='b', linestyle='--', label="Mean:" + str(np.round(np.mean(before_rad), decimals=2)))
    ax.hist(after_rad, bins=bins, alpha=1, color='r', linewidth=2, histtype='step', label='After linking', density=False)
    ax.axvline(np.mean(after_rad), color='r', linestyle='--', label="Mean:" + str(np.round(np.mean(after_rad), decimals=2)))

    plt.xlabel("Radius")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #attenuation
    min_val = min(np.min(before_att), np.min(after_att))
    max_val = max(np.max(before_att), np.max(after_att))
    bins = np.linspace(min_val, max_val, num_bins)

    plt.figure()
    ax = plt.gca()
    ax.hist(before_att, bins=bins, alpha=1, color='b', linewidth=2, histtype='step', label='Before linking',
            density=False)
    ax.axvline(np.mean(before_att), color='b', linestyle='--',
               label="Mean:" + str(np.round(np.mean(before_att), decimals=2)))
    ax.hist(after_att, bins=bins, alpha=1, color='r', linewidth=2, histtype='step', label='After linking', density=False)
    ax.axvline(np.mean(after_att), color='r', linestyle='--',
               label="Mean:" + str(np.round(np.mean(after_att), decimals=2)))

    plt.xlabel("Attenuation")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.show()


def scale_vectors_to_target(source_vecs, target_vecs, upper_pct=99.0):
    """
    Scales a set of N,3 source vectors (e.g., simulation) to match the
    magnitude scale of target vectors (e.g., experiment).
    """
    # 1. Calculate the magnitudes (L2 norm) of the N,3 vectors
    source_mags = np.linalg.norm(source_vecs, axis=1)
    target_mags = np.linalg.norm(target_vecs, axis=1)

    # Filter out zero-magnitude vectors before finding the percentile
    source_mags_valid = source_mags[source_mags > 0]
    target_mags_valid = target_mags[target_mags > 0]

    # 2. Find the robust "peak" magnitudes using the given percentile
    source_peak = np.nanpercentile(source_mags_valid, upper_pct)
    target_peak = np.nanpercentile(target_mags_valid, upper_pct)

    # 3. Calculate the global scaling factor (tiny epsilon avoids division by zero)
    scale_factor = target_peak / (source_peak + 1e-8)

    # 4. Scale the source vectors
    scaled_source_vecs = source_vecs * scale_factor

    return scaled_source_vecs, scale_factor


def load_and_filter_data(cfg):
    """Load the porous-sample TrackPy and CTracks results, filter/link CTracks, and print separation stats.

    Returns:
        (mask, ctracks_files, dataframe, tp_data, trackpy_data, ctracks_data), where `trackpy_data`
        and `ctracks_data` are (pos, vel, pos_all, vel_all, vel_all_magnitude) tuples.
    """
    mask = tifffile.imread(cfg['mask_file']) > SAMPLE_GEOMETRY['mask_threshold']
    mask = mask[SAMPLE_GEOMETRY['roi_height'], SAMPLE_GEOMETRY['roi_width'], SAMPLE_GEOMETRY['roi_width']]
    mask = mask.astype(np.uint8)

    # load trackpy results
    tp_data = pd.read_csv(cfg['trackpy_recon_file'] + ".csv")  # dataframe particle, frame, z,y,x,vz,vy,vx,velMags
    tp_data.drop(index=0, inplace=True)
    trackpy_pos, trackpy_vel, trackpy_pos_all, trackpy_vel_all, trackpy_vel_all_magnitude = convert_trackpy_to_frame_lists(
        tp_data, frame_range=cfg['ctracks_frame_range'])
    print(f"Loaded {trackpy_pos_all.shape[0] / len(cfg['ctracks_frame_range'])} average trackpy tracks per frame")

    # load ctrack results
    ctracks_files = filelist(cfg['ctracks_recon_dir'], fmt="scan_%s.npy", sort=True)[
        cfg['ctracks_frame_range'][0]:cfg['ctracks_frame_range'][-1]]
    if not RUN_OPTIONS['load_dataframe']:
        ctracks_pos, ctracks_vel, ctracks_pos_all, ctracks_vel_all, ctracks_vel_all_magnitude, ctracks_shape = \
            load_ctrack_files(ctracks_files)

        dataframe, (ctracks_pos_analysis, ctracks_vel_analysis, ctracks_pos_all, ctracks_vel_all,
                    ctracks_vel_all_magnitude) = filter_vectors_direct(
            ctracks_pos, ctracks_vel, ctracks_shape, **cfg['filter_params'])
        dataframe.to_csv(os.path.join(cfg['ctracks_recon_dir'], "linked_trajectories.csv"), index=False)
    else:
        dataframe = pd.read_csv(os.path.join(cfg['ctracks_recon_dir'], "linked_trajectories.csv"))
        ctracks_pos_analysis, ctracks_vel_analysis, ctracks_pos_all, ctracks_vel_all, ctracks_vel_all_magnitude = \
            convert_trackpy_to_frame_lists(dataframe)

    print(f"Loaded {ctracks_pos_all.shape[0] / len(ctracks_files)} average ctrack tracks per frame")

    print(f"Found {np.sum(ctracks_vel_all_magnitude > DETECTION_VELOCITY_THRESHOLD) / len(cfg['ctracks_frame_range'])} "
          f"average ctracks tracks above {DETECTION_VELOCITY_THRESHOLD} per frame")
    print(f"Found {np.sum(trackpy_vel_all_magnitude > DETECTION_VELOCITY_THRESHOLD) / len(ctracks_files)} "
          f"average trackpy tracks above {DETECTION_VELOCITY_THRESHOLD} per frame")

    print("CTracks - mean separation")
    calc_mean_distances(ctracks_pos_analysis)
    print("TrackPy - mean separation")
    calc_mean_distances(trackpy_pos)

    trackpy_data = (trackpy_pos, trackpy_vel, trackpy_pos_all, trackpy_vel_all, trackpy_vel_all_magnitude)
    ctracks_data = (ctracks_pos_analysis, ctracks_vel_analysis, ctracks_pos_all, ctracks_vel_all, ctracks_vel_all_magnitude)
    return mask, ctracks_files, dataframe, tp_data, trackpy_data, ctracks_data


def plot_velocity_distribution_comparisons(cfg, ctracks_vel_all, trackpy_vel_all):
    """Plot the CTracks-vs-TrackPy velocity distribution, and optionally vs. simulated ground truth."""
    result_dir = analysis_results_dir(cfg)
    plot_velocity_distributions(ctracks_vel_all, trackpy_vel_all,
                                save_file=os.path.join(result_dir, f"velocity_distribution.{FIGURE_EXT}"))

    if RUN_OPTIONS['compare_simulation_distribution']:
        sim_substeps_per_frame = ACQUISITION_PARAMS['sim_substeps_per_frame']
        gt_tracks = np.load(GROUND_TRUTH_TRACKS_FILE, allow_pickle=True)  # N,851*7,3
        gt_vel = np.gradient(gt_tracks, 1 / sim_substeps_per_frame, axis=1, edge_order=1)
        gt_vel = np.concatenate(
            gt_vel.reshape(gt_vel.shape[0], gt_vel.shape[1] // sim_substeps_per_frame, sim_substeps_per_frame, 3).mean(axis=2),
            axis=0)
        gt_vel, scale_factor = scale_vectors_to_target(gt_vel, ctracks_vel_all)
        velocities_list = [gt_vel, ctracks_vel_all, trackpy_vel_all]
        plot_velocity_distributions_general(velocities_list, ['Simulation', 'CTracks', 'TrackPy'], num_bins=20,
                                            save_file=os.path.join(result_dir, f"velocity_distributions_simulation.{FIGURE_EXT}"),
                                            colors=['green', 'blue', 'red'], log_y=False)


def _plot_simulation_field_comparison(cfg, plot_mask, slice_vals, trackpy_field, ctracks_field):
    """Compare experimental velocity fields to the simulated ground-truth field via log-normalized metrics/plots."""
    sim_field = tifffile.imread(SIM_FIELD_FILE)
    sim_field = zoom(sim_field, zoom=MASK_ZOOM_FACTOR)

    def log_normalize(field, upper_pct=SIM_FIELD_LOG_PERCENTILE, shared_peak=None):
        """Log1p-normalize a field to [0, 1] using either its own or a shared percentile peak."""
        mask = field > 0
        # log1p handles small values gracefully
        log_field = np.zeros_like(field, dtype=float)
        log_field[mask] = np.log1p(field[mask])

        # Use the provided shared peak, or calculate it locally if none is provided
        peak = np.nanpercentile(log_field[mask], upper_pct) if shared_peak is None else shared_peak

        norm_field = log_field / (peak + 1e-8)
        norm_field = np.clip(norm_field, 0, 1)
        return norm_field

    # Combine the valid (positive) data from both experimental fields to get a shared peak
    tp_valid = trackpy_field[trackpy_field > 0]
    ctracks_valid = ctracks_field[ctracks_field > 0]
    combined_exp_data = np.concatenate([tp_valid, ctracks_valid])
    shared_log_peak = np.nanpercentile(np.log1p(combined_exp_data), SIM_FIELD_LOG_PERCENTILE)

    sim_norm = log_normalize(sim_field)
    tp_norm = log_normalize(trackpy_field, shared_peak=shared_log_peak)
    ctracks_norm = log_normalize(ctracks_field, shared_peak=shared_log_peak)

    calculate_and_print_metrics_comparison(sim_norm, tp_norm, ctracks_norm, pore_mask=plot_mask,
                                           method_1_name="RDL", method_2_name="CTracks")

    trackpy_sim_diff = sim_norm - tp_norm
    ctracks_sim_diff = sim_norm - ctracks_norm
    trackpy_sim_diff = np.abs(trackpy_sim_diff)
    ctracks_sim_diff = np.abs(ctracks_sim_diff)

    result_dir = analysis_results_dir(cfg)
    plot_velocity_field_slice(plot_mask[:, slice_vals[1] + cfg['slice_shift'], :],
                              trackpy_sim_diff[:, slice_vals[1] + cfg['slice_shift'], :],
                              v_range=DIFF_FIELD_VRANGE,
                              file_loc=os.path.join(result_dir, f"sim_trackpy_field-difference.{FIGURE_EXT}"),
                              cmap="viridis")
    plot_velocity_field_slice(plot_mask[:, slice_vals[1] + cfg['slice_shift'], :],
                          ctracks_sim_diff[:, slice_vals[1] + cfg['slice_shift'], :],
                          v_range=DIFF_FIELD_VRANGE,
                          file_loc=os.path.join(result_dir, f"sim_ctracks_field-difference.{FIGURE_EXT}"),
                          cmap="viridis")

    trackpy_sim_diff_abs = np.abs(trackpy_sim_diff)
    ctracks_sim_diff_abs = np.abs(ctracks_sim_diff)
    plot_velocity_field_slice(plot_mask[:, slice_vals[1] + cfg['slice_shift'], :],
                              trackpy_sim_diff_abs[:, slice_vals[1] + cfg['slice_shift'], :],
                              v_range=DIFF_FIELD_ABS_VRANGE,
                              file_loc=os.path.join(result_dir, f"sim_trackpy_field-difference_abs.{FIGURE_EXT}"))
    plot_velocity_field_slice(plot_mask[:, slice_vals[1] + cfg['slice_shift'], :],
                              ctracks_sim_diff_abs[:, slice_vals[1] + cfg['slice_shift'], :],
                              v_range=DIFF_FIELD_ABS_VRANGE,
                              file_loc=os.path.join(result_dir, f"sim_ctracks_field-difference_abs.{FIGURE_EXT}"))


def plot_velocity_fields(cfg, mask):
    """Plot experimental velocity-field slice comparisons, and optionally vs. the simulated field."""
    field_folder = os.path.join(cfg['basefolder'], "interp_fields")

    # interp_fields tifs are (Z, 3, Y, X) velocity vector fields, not magnitudes - reduce
    # over the component axis (axis=1) to get the (Z, Y, X) magnitude volumes this function needs.
    trackpy_vec = tifffile.imread(os.path.join(field_folder, "trackpy_interp_field.tif"))
    ctracks_vec = tifffile.imread(os.path.join(field_folder, "ctracks_interp_field.tif"))
    trackpy_field = np.linalg.norm(trackpy_vec, axis=1)
    ctracks_field = np.linalg.norm(ctracks_vec, axis=1)
    diff_field = ctracks_field - trackpy_field

    plot_mask = zoom(mask, zoom=MASK_ZOOM_FACTOR, order=0)
    slice_vals = np.array(plot_mask.shape) // 2
    result_dir = analysis_results_dir(cfg)

    # plot ctracks trackpy side by side
    plot_two_velocity_slices(plot_mask, [trackpy_field, ctracks_field], cfg['interp_vrange'], ['TrackPy', 'CTracks'],
                             file_loc=os.path.join(result_dir, f"field-sidebyside.{FIGURE_EXT}"), figsize=(12, 10),
                             slice_shift=cfg['slice_shift'])

    # plot each separately as well
    plot_velocity_field_slice(plot_mask[:, slice_vals[1] + cfg['slice_shift'], :],
                              trackpy_field[:, slice_vals[1] + cfg['slice_shift'], :],
                              v_range=cfg['interp_vrange'],
                              file_loc=os.path.join(result_dir, f"field-trackpy.{FIGURE_EXT}"),
                              cmap="viridis")

    plot_velocity_field_slice(plot_mask[:, slice_vals[1] + cfg['slice_shift'], :],
                              ctracks_field[:, slice_vals[1] + cfg['slice_shift'], :],
                              v_range=cfg['interp_vrange'],
                              file_loc=os.path.join(result_dir, f"field-ctracks.{FIGURE_EXT}"),
                              cmap="viridis")

    # plot difference alone
    plot_velocity_field_slice(plot_mask[:, slice_vals[1] + cfg['slice_shift'], :],
                              diff_field[:, slice_vals[1] + cfg['slice_shift'], :],
                              v_range=(-cfg['interp_vrange'][1] / 2, cfg['interp_vrange'][1] / 2),
                              file_loc=os.path.join(result_dir, f"field-difference.{FIGURE_EXT}"), cmap="seismic")

    if RUN_OPTIONS['compare_simulation_field']:
        _plot_simulation_field_comparison(cfg, plot_mask, slice_vals, trackpy_field, ctracks_field)


def plot_2d_annotated_slices(cfg, dataframe, tp_data, crop_correction_ctracks):
    """Overlay annotated 2D slice plots of the linked CTracks trajectories and TrackPy detections."""
    dataframe[['x', 'y', 'z']] += crop_correction_ctracks
    slice_save_folder = os.path.join(analysis_results_dir(cfg), "annotated_slices")
    plot_slices(dataframe, cfg['recon_folder'], IMAGE_FOLDER_BASE, SAMPLE_GEOMETRY['frame_crop'], tp_data, slice_save_folder)


def build_plot_style_params(cfg):
    """Build the 3D-plot styling params (crop/colour-limits/velocity split) for this dataset."""
    return {
        "scale": 1,
        "down_sample_factor": 4,
        "crop_bounds": cfg['plot_crop'],
        "arrow_lim": (0.5, 1),
        "plot_grains": True,
        "show_arrowhead": True,
        "surface_transparency": 0.2,
        "clim": cfg['clim'],
        "velocity_split": cfg['velocity_split'],
        "plot_low": True,
        "interactive": False,
    }


def plot_3d_visualizations(cfg, crop_correction_ctracks):
    """Render 3D streamline plots and/or vector videos for the CTracks and TrackPy results."""
    shift = crop_correction_ctracks
    default_params = build_plot_style_params(cfg)
    result_dir = analysis_results_dir(cfg)

    if RUN_OPTIONS['plot_3D_streamlines']:
        # plot ctracks
        plot_single(SEG_PATH_CROPPED, os.path.join(cfg['ctracks_recon_dir'], "linked_trajectories.csv"), shift=shift,
                    save_file=os.path.join(result_dir, "ctracks_streamlines.png"), cam_params=CAM_PARAMS,
                    cbar_params=CBAR_PARAMS, **default_params)
        # plot trackpy
        plot_single(SEG_PATH_CROPPED, cfg['trackpy_recon_file'] + ".csv", shift=np.zeros(3),
                    save_file=os.path.join(result_dir, "trackpy_streamlines.png"), cam_params=CAM_PARAMS,
                    cbar_params=CBAR_PARAMS, **default_params)
    if RUN_OPTIONS['plot_3D_video']:
        # plot videos of vectors
        plot_video(SEG_PATH_CROPPED, os.path.join(cfg['ctracks_recon_dir'], "linked_trajectories.csv"), shift=shift,
                    save_folder=os.path.join(result_dir, "ctracks_vector_video"), cam_params=CAM_PARAMS,
                    cbar_params=CBAR_PARAMS, **default_params)

        plot_video(SEG_PATH_CROPPED, cfg['trackpy_recon_file'] + ".csv", shift=np.zeros(3),
                    save_folder=os.path.join(result_dir, "trackpy_vector_video"), cam_params=CAM_PARAMS,
                    cbar_params=CBAR_PARAMS, **default_params)


def main():
    """Analyse porous-sample particle tracks: filter/link, compare to simulation, and plot results."""
    cfg = DATASETS[ACTIVE_DATASET]

    mask, ctracks_files, dataframe, tp_data, trackpy_data, ctracks_data = load_and_filter_data(cfg)
    _, _, _, trackpy_vel_all, _ = trackpy_data
    _, _, _, ctracks_vel_all, _ = ctracks_data

    plot_velocity_distribution_comparisons(cfg, ctracks_vel_all, trackpy_vel_all)

    if RUN_OPTIONS['plot_velfields']:
        plot_velocity_fields(cfg, mask)

    frame_crop = SAMPLE_GEOMETRY['frame_crop']
    crop_correction_ctracks = -1 * np.array([frame_crop[2][0], frame_crop[1][0], frame_crop[0][0]])

    if RUN_OPTIONS['plot_2D_slices']:
        plot_2d_annotated_slices(cfg, dataframe, tp_data, crop_correction_ctracks)

    plot_3d_visualizations(cfg, crop_correction_ctracks)


if __name__ == "__main__":
    main()
