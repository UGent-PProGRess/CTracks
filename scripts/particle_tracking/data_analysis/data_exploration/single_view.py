"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

Single-panel pyvista 3D viewer for particle tracks: interactively scrub through frames
of either a linked-trajectories CSV (TrackPy/CTracks format) or raw ctracks
`scan_*.npy`-style reconstruction files, overlaid on the sample's pore-mask surface.

Replaces the former vis_3d_experimental.py and 3d_explore.py demos, which duplicated
almost all of this logic - the shared engine now lives in
`vis3D/plot_3d_vectors.py` (see `load_particles_df`, `load_particles_ctracks`,
`plot_interactive`). See `duo_view.py` for the side-by-side comparison equivalent.

Not imported anywhere else in the repo - standalone exploration script.
"""

import os

import numpy as np

from scripts.particle_tracking.data_analysis.vis3D.plot_3d_vectors import (
    load_particles_df, load_particles_ctracks, plot_interactive,
)

# ============================================================================
# User-adjustable configuration
# ============================================================================

# Data root - change this if CTracksSubmissionData is moved or mounted under a
# different drive/path (e.g. moving to a new PC).
DATA_ROOT = "J:\\"
_POROUS_GLASS_DIR = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Porous glass")
_SIMULATED_DIR = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Simulated")

# Demo-block dataset configs: each picks a data source ("df" = linked-trajectories CSV,
# "ctracks" = raw scan_*.npy reconstruction files) plus the pore-mask/geometry to view
# it against.
DATASETS = {
    "porous_180nlmin_trackpy": dict(
        source="df",
        particle_file=os.path.join(_POROUS_GLASS_DIR, r"180nlmin\rdl_results\visco_sample2_180nlmin_velocityPoints.csv"),
        seg_path=os.path.join(_POROUS_GLASS_DIR, "segmentation_cropped.tif"),
        shift=np.array((-80, -80, -224)),
        plot_all_tracks=True,
        crop_bounds=[50, 250, 50, 250, 50, 250],
        clim=(1, 15),
        arrow_lim=(0.5, 1),
        velocity_split=4,
    ),
    "simulation_ctracks_reconstruction": dict(
        source="ctracks",
        track_set="reconstruction",  # or "ground truth"
        particle_file=os.path.join(_SIMULATED_DIR, r"ctracks_trajectories\params_rot*.npy"),
        seg_path=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Simulation\fields\mask.tif"),
        shift=np.array((0, 0, 0)),
        plot_all_tracks=False,
        crop_bounds=[100, 200, 100, 200, 100, 200],
        clim=(0, 20),
        arrow_lim=(1, 2),
        velocity_split=5,
    ),
}
ACTIVE_DATASET = "porous_180nlmin_trackpy"

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
SHOW_ARROWHEAD = True
PLOT_LOW = True


def main():
    """Demo: interactively scrub (or save as a GIF) through one dataset's particle tracks
    against its pore-mask surface."""
    cfg = DATASETS[ACTIVE_DATASET]

    if cfg["source"] == "df":
        iterators = load_particles_df(cfg["particle_file"], cfg["shift"], arrow_lim=cfg["arrow_lim"],
                                      plot_all_tracks=cfg["plot_all_tracks"], velocity_split=cfg["velocity_split"],
                                      show_arrowhead=SHOW_ARROWHEAD, plot_low=PLOT_LOW)
    else:
        iterators = load_particles_ctracks(cfg["particle_file"], track_set=cfg["track_set"], total_shift=cfg["shift"],
                                           arrow_lim=cfg["arrow_lim"], plot_all_tracks=cfg["plot_all_tracks"],
                                           velocity_split=cfg["velocity_split"], show_arrowhead=SHOW_ARROWHEAD,
                                           plot_low=PLOT_LOW)

    plot_interactive(cfg["seg_path"], iterators, scale=SCALE, down_sample_factor=DOWN_SAMPLE_FACTOR,
                     crop_bounds=cfg["crop_bounds"], clim=cfg["clim"], show_clip_panel=SHOW_CLIP_PANEL,
                     save_fig=SAVE_FIG, move_camera=MOVE_CAMERA, savefile=SAVE_PATH,
                     plot_grains=PLOT_GRAINS, surface_transparency=SURFACE_TRANSPARENCY)


if __name__ == "__main__":
    main()
