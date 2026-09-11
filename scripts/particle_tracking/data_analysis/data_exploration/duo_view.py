"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

Dual-panel pyvista 3D viewer: two linked side-by-side views for comparing particle
tracks - either TrackPy vs CTracks (experimental porous-media datasets) or ground
truth vs reconstruction (simulated dataset).

Replaces the former duo_view_experimental.py and duo_view_simulation.py demos, which
duplicated almost all of this logic - the shared engine now lives in
`vis3D/plot_3d_vectors.py` (see `load_particles_df`, `load_particles_ctracks_dual`,
`plot_duo`). See `single_view.py` for the single-panel equivalent.

Not imported anywhere else in the repo - standalone exploration script.
"""

import os

import numpy as np

from scripts.particle_tracking.data_analysis.vis3D.plot_3d_vectors import (
    load_particles_df, load_particles_ctracks_dual, plot_duo,
)

# ============================================================================
# User-adjustable configuration
# ============================================================================

# Data root - change this if CTracksSubmissionData is moved or mounted under a
# different drive/path (e.g. moving to a new PC).
DATA_ROOT = "J:\\"
_POROUS_GLASS_DIR = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Porous glass")
_SIMULATED_DIR = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Simulated")
# Shared cropped segmentation mask - one file for all experimental datasets below.
_SEG_PATH_CROPPED = os.path.join(_POROUS_GLASS_DIR, "segmentation_cropped.tif")

# Demo-block dataset configs: each is either an "experimental" (TrackPy vs CTracks,
# left_file/right_file) or "simulation" (ground truth vs reconstruction, particle_file
# with both track sets read from the same files) comparison. `labels` gives the
# (left, right) panel titles.
DATASETS = {
    "porous_60nlmin": dict(
        source="experimental",
        seg_path=_SEG_PATH_CROPPED,
        left_file=os.path.join(_POROUS_GLASS_DIR, r"60nlmin\rdl_results\visco_sample2_60nlmin_velocityPoints.csv"),
        right_file=os.path.join(_POROUS_GLASS_DIR, r"60nlmin\ctracks_trajectories\linked_trajectories.csv"),
        labels=("TrackPy", "CTracks"),
        shift=np.array((-80, -80, -224)),
        crop_bounds=[50, 250, 50, 250, 50, 250],
        clim=(1, 15),
    ),
    "porous_180nlmin": dict(
        source="experimental",
        seg_path=_SEG_PATH_CROPPED,
        left_file=os.path.join(_POROUS_GLASS_DIR, r"180nlmin\rdl_results\visco_sample2_180nlmin_velocityPoints.csv"),
        right_file=os.path.join(_POROUS_GLASS_DIR, r"180nlmin\ctracks_trajectories\linked_trajectories.csv"),
        labels=("TrackPy", "CTracks"),
        shift=np.array((-80, -80, -224)),
        crop_bounds=[50, 250, 50, 250, 50, 250],
        clim=(1, 15),
    ),
    "porous_1ulmin": dict(
        source="experimental",
        seg_path=_SEG_PATH_CROPPED,
        left_file=os.path.join(_POROUS_GLASS_DIR, r"1ulmin\rdl_results\visco_sample2_1ulmin_velocityPoints.csv"),
        right_file=os.path.join(_POROUS_GLASS_DIR, r"1ulmin\ctracks_trajectories\linked_trajectories.csv"),
        labels=("TrackPy", "CTracks"),
        shift=np.array((-80, -80, -224)),
        crop_bounds=[50, 250, 50, 250, 50, 250],
        clim=(1, 15),
    ),
    "simulation_ground_truth_vs_recon": dict(
        source="simulation",
        seg_path=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Simulation\fields\mask.tif"),
        particle_file=os.path.join(_SIMULATED_DIR, r"ctracks_trajectories\params_rot*.npy"),
        labels=("Ground Truth", "Reconstruction"),
        shift=np.array((0, 0, 0)),
        crop_bounds=[100, 200, 100, 200, 100, 200],
        clim=(20, 50),
    ),
}
ACTIVE_DATASET = "porous_180nlmin"

# Whether to save the scrubbed animation as a GIF (to SAVE_PATH) instead of opening an
# interactive window.
SAVE_FIG = False
SAVE_PATH = "./animation.gif"
MOVE_CAMERA = False

SCALE = 1
DOWN_SAMPLE_FACTOR = 4
PLOT_GRAINS = True
SURFACE_TRANSPARENCY = 0.2
SHOW_CLIP_PANEL = False
PLOT_ALL_TRACKS = True
SHOW_ARROWHEAD = True
ARROW_LIM = (0.5, 1)
VELOCITY_SPLIT = 4
PLOT_LOW = True
# Matches the original duo-view scripts' hardcoded camera (not `plot_3d_vectors.py`'s
# single-panel default).
CAM_PARAMS = {"azimuth": -30, "elevation": 15, "focal_shift": [0, 0, 0], "zoom": 1}


def main():
    """Demo: compare two particle-track sources for the active dataset in a linked dual 3D view."""
    cfg = DATASETS[ACTIVE_DATASET]

    if cfg["source"] == "simulation":
        # Both panels read the same files; "ground truth"/"reconstruction" are two
        # different fields stored per file, matching labels=("Ground Truth", "Reconstruction").
        left_iterators, right_iterators = load_particles_ctracks_dual(
            cfg["particle_file"], total_shift=cfg["shift"], arrow_lim=ARROW_LIM,
            plot_all_tracks=PLOT_ALL_TRACKS, velocity_split=VELOCITY_SPLIT,
            show_arrowhead=SHOW_ARROWHEAD, plot_low=PLOT_LOW)
    else:
        # Right panel (CTracks) is loaded first, with its own shift and no frame cap;
        # left panel (TrackPy) is loaded second, zero shift, capped to the right panel's
        # frame count - matching the original duo_view_experimental.py behaviour.
        right_iterators = load_particles_df(cfg["right_file"], cfg["shift"], arrow_lim=ARROW_LIM,
                                            plot_all_tracks=PLOT_ALL_TRACKS, velocity_split=VELOCITY_SPLIT,
                                            show_arrowhead=SHOW_ARROWHEAD, plot_low=PLOT_LOW)
        left_iterators = load_particles_df(cfg["left_file"], np.zeros(3), arrow_lim=ARROW_LIM,
                                           plot_all_tracks=PLOT_ALL_TRACKS, velocity_split=VELOCITY_SPLIT,
                                           show_arrowhead=SHOW_ARROWHEAD, plot_low=PLOT_LOW,
                                           frame_end=right_iterators[1])

    plot_duo(cfg["seg_path"], left_iterators, right_iterators, scale=SCALE,
            down_sample_factor=DOWN_SAMPLE_FACTOR, crop_bounds=cfg["crop_bounds"], clim=cfg["clim"],
            show_clip_panel=SHOW_CLIP_PANEL, save_fig=SAVE_FIG, move_camera=MOVE_CAMERA,
            savefile=SAVE_PATH, plot_grains=PLOT_GRAINS, surface_transparency=SURFACE_TRANSPARENCY,
            labels=cfg["labels"], cam_params=CAM_PARAMS)


if __name__ == "__main__":
    main()
