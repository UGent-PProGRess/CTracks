"""Overlay linked particle trajectories onto reconstruction slice images.

Unlike `plot_slices.py` (a standalone script that just saves raw orthogonal
slices), this module's `plot_slices` is a real importable entry point - it is
used by `analyse_porous.py` to save one annotated mid-y slice per frame, with
the linked-trajectory detections (and, optionally, a second `trackpy`-style
dataset for comparison) scattered on top via `plotTrajectoriesOnSlice`.
`make_colorbar` renders the matching standalone colorbar legend for the
velocity-magnitude coloring `plotTrajectoriesOnSlice` uses. `main` is a
demo/config block (see `DEMO_DATASETS`) exercising `make_colorbar` only, with
the equivalent `plot_slices` call left commented, as in the original script.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pims
from scripts.particle_tracking.data_analysis.vis2D import stackReader as sr


def make_colorbar(save_file = "velocity_colorbar.png"):
    """Render and save a standalone horizontal colorbar for a 0-1 normalized
    velocity-magnitude scale (matching the `color_mode='velocity'` coloring
    used by `plotTrajectoriesOnSlice`), for use as a figure legend.
    """
    fig_cb, ax_cb = plt.subplots(figsize=(6, 0.5))

    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cmap = mpl.cm.viridis

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    cb = fig_cb.colorbar(mappable, cax=ax_cb, orientation='horizontal')
    cb.set_label('Normalized Velocity Magnitude', fontsize=18)
    cb.ax.set_xticklabels([])

    fig_cb.savefig(save_file, dpi=300, bbox_inches='tight')

def plotTrajectoriesOnSlice(trajectories_filtered, ax, yDisplay, sliceThickness=5, color_mode='velocity', min_size=50,
                            max_size=90):
    """Scatter-plot particle detections within `yDisplay +/- sliceThickness` onto `ax`.

    Marker color encodes either per-slice-normalized velocity magnitude
    (`color_mode='velocity'`), particle ID (`color_mode='particle'`, colored
    mod 20 via tab20), or a fixed color/array (any other `color_mode` value).
    Marker size is linearly scaled by distance from the slice center, so
    particles closer to `yDisplay` are drawn larger.

    Returns the `PathCollection` from the filled-marker scatter (usable as a
    colorbar mappable).
    """
    # Filter particles within the slice
    mask = (trajectories_filtered['y'].values > yDisplay - sliceThickness) & \
           (trajectories_filtered['y'].values < yDisplay + sliceThickness)
    particlesToPlot = trajectories_filtered[mask]

    # Extract positions
    x = particlesToPlot['x'].to_numpy()
    z = particlesToPlot['z'].to_numpy()
    y = particlesToPlot['y'].to_numpy()

    # 1. Determine Colors
    if color_mode == 'velocity':
        vx = particlesToPlot['vx'].to_numpy()
        vy = particlesToPlot['vy'].to_numpy()
        vz = particlesToPlot['vz'].to_numpy()

        # Calculate raw velocity magnitude
        v_mag = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        v_min = v_mag.min()
        v_max = v_mag.max()

        # Normalize to 0-1 for this specific slice
        if v_max > v_min:
            colors = (v_mag - v_min) / (v_max - v_min)
        else:
            # Fallback in case all particles happen to have the exact same speed
            colors = np.zeros_like(v_mag)

        vmin, vmax = 0, 1  # Hardcode limits to 0 and 1
        cmap = mpl.cm.viridis

    elif color_mode == 'particle':
        colors = particlesToPlot["particle"].to_numpy() % 20
        cmap = "tab20"
        vmin, vmax = 0, 19
    else:
        colors = color_mode
        cmap = None
        vmin, vmax = None, None

    # 2. Calculate Sizes based on distance to yDisplay
    dist_to_center = np.abs(y - yDisplay)
    norm_dist = dist_to_center / sliceThickness
    sizes = max_size - (norm_dist * (max_size - min_size))
    # sizes = max_size
    outline_sizes = sizes * 1.5

    # Plot filled circles
    bo = ax.scatter(x=x, y=z, s=sizes, marker="o", c=colors, cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax)

    # Plot outlines
    ax.scatter(x=x, y=z, s=outline_sizes, marker=mpl.markers.MarkerStyle(marker="o", fillstyle='none'),
               c=colors, cmap=cmap, vmin=vmin, vmax=vmax)

    return bo

def plot_slices(ctracks_df, recon_folder, imageFolderBase = "recon_", crop_params =  None, trackpy_df = None, save_folder = "./"):
    """Save one annotated horizontal (xz) slice image per frame in `ctracks_df`.

    For each frame present in `ctracks_df`, the reconstruction slab is read
    via `stackReader`/`pims` from `recon_folder`, cropped by `crop_params`,
    and its mid-y slice is plotted with the linked `ctracks_df` detections
    (and, if given, `trackpy_df` detections) overlaid via
    `plotTrajectoriesOnSlice`. Images are written to
    `save_folder/annotated_<frame>.png`.

    Parameters
    ----------
    ctracks_df : DataFrame
        Linked particle trajectories with columns x, y, z, vx, vy, vz, frame.
    recon_folder : str
        Folder containing the per-frame reconstruction subfolders read by `stackReader`.
    imageFolderBase : str
        Substring identifying reconstruction subfolders within `recon_folder` (see `stackReader`).
    crop_params : tuple or None
        Crop margins passed to `pims.process.crop`, as ((z0,z1),(y0,y1),(x0,x1)).
    trackpy_df : DataFrame or None
        Optional second set of trajectories (e.g. from trackpy) overlaid for comparison.
    save_folder : str
        Output directory for the annotated slice images.
    """
    os.makedirs(save_folder, exist_ok=True)

    frame_numbers = ctracks_df.frame.unique()
    # frame_numbers = range(0,7)

    frames = sr.set_pipeline(sr.stackReader(superfolder=recon_folder, imageFolderBase=imageFolderBase))
    frames = pims.process.crop(frames, crop_params)

    # loop over frames
    for frame_num in frame_numbers:
        print(f"Saving frame {frame_num}")
        frame_image = frames[frame_num]
        fifi, ax = plt.subplots(figsize=(18, 12), dpi=80)
        yDisplay = frame_image.shape[1] // 2
        frame_slice = frame_image[(slice(None), yDisplay, slice(None))]
        ax.set_xlim(-0.5, (frame_slice.shape[1] - 0.5))
        ax.set_ylim(-0.5, (frame_slice.shape[0] - 0.5))
        plt.imshow(frame_slice, cmap='gray')

        frame_ctracks = ctracks_df[ctracks_df['frame'] == frame_num]

        plotTrajectoriesOnSlice(frame_ctracks, ax, yDisplay, sliceThickness = 5)
        if trackpy_df is not None:
            frame_trackpy = trackpy_df[trackpy_df['frame'] == frame_num]
            plotTrajectoriesOnSlice(frame_trackpy, ax, yDisplay, sliceThickness = 3, color_mode='particle')
        plt.savefig(os.path.join(save_folder, f"annotated_{frame_num}.png"))
        plt.close(fifi)

# Data root - change this if CTracksSubmissionData is moved or mounted under a
# different drive/path (e.g. moving to a new PC).
DATA_ROOT = "J:\\"
_CTSD_PROCESSED = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental")
_CTSD_RAW = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental")

# Demo-block dataset configs (see `main` below). crop_params is
# (z, y, x) margins; ctracks_z_correction offsets for a recon that was
# cropped further in z than the trackpy comparison data.
DEMO_DATASETS = {
    "capillary_60nlmin": dict(
        basedir=os.path.join(_CTSD_PROCESSED, "Capillary"),
        recon_folder=os.path.join(_CTSD_RAW, r"Capillary\60nlmin\recon"),
        basefolder_name="60nlmin",
        trackpy_file_rel=r"rdl_results\capillary_60nlmin_velocityPoints.csv",
        crop_params=((0, 148), (100, 110), (100, 110)),  # trackpy cropped, ctracks uncropped
        ctracks_z_correction=0,
    ),
    "capillary_180nlmin": dict(
        basedir=os.path.join(_CTSD_PROCESSED, "Capillary"),
        recon_folder=os.path.join(_CTSD_RAW, r"Capillary\180nlmin\recon"),
        basefolder_name="180nlmin",
        trackpy_file_rel=r"rdl_results\capillary_180nlmin_velocityPoints.csv",
        crop_params=((0, 148), (100, 110), (100, 110)),  # trackpy cropped, ctracks uncropped
        ctracks_z_correction=0,
    ),
    "capillary_1ulmin": dict(
        basedir=os.path.join(_CTSD_PROCESSED, "Capillary"),
        recon_folder=os.path.join(_CTSD_RAW, r"Capillary\1ulmin\recon"),
        basefolder_name="1ulmin",
        trackpy_file_rel=r"rdl_results\capillary_1ulmin_velocityPoints.csv",
        crop_params=((0, 148), (100, 110), (100, 110)),
        ctracks_z_correction=57,  # recon was cropped extra
    ),
    "porous_60nlmin": dict(
        basedir=os.path.join(_CTSD_PROCESSED, "Porous glass"),
        recon_folder=os.path.join(_CTSD_RAW, r"Porous glass\60nlmin\recon"),
        basefolder_name="60nlmin",
        trackpy_file_rel=r"rdl_results\visco_sample2_60nlmin_velocityPoints.csv",
        crop_params=((224, 224), (80, 80), (80, 80)),
        ctracks_z_correction=0,
    ),
    "porous_180nlmin": dict(
        basedir=os.path.join(_CTSD_PROCESSED, "Porous glass"),
        recon_folder=os.path.join(_CTSD_RAW, r"Porous glass\180nlmin\recon"),
        basefolder_name="180nlmin",
        trackpy_file_rel=r"rdl_results\visco_sample2_180nlmin_velocityPoints.csv",
        crop_params=((224, 224), (80, 80), (80, 80)),
        ctracks_z_correction=0,
    ),
    "porous_1ulmin": dict(
        basedir=os.path.join(_CTSD_PROCESSED, "Porous glass"),
        recon_folder=os.path.join(_CTSD_RAW, r"Porous glass\1ulmin\recon"),
        basefolder_name="1ulmin",
        trackpy_file_rel=r"rdl_results\visco_sample2_1ulmin_velocityPoints.csv",
        crop_params=((224, 224), (80, 80), (80, 80)),
        ctracks_z_correction=0,
    ),
}
ACTIVE_DEMO_DATASET = "capillary_1ulmin"


def main():
    """Demo: build the ctracks dataframe for the active dataset (applying the
    crop correction) and save its velocity colorbar legend. The actual
    `plot_slices` call is left commented, as in the original script - this
    block otherwise only exercises `make_colorbar`.
    """
    cfg = DEMO_DATASETS[ACTIVE_DEMO_DATASET]
    basedir = cfg["basedir"]
    recon_folder = cfg["recon_folder"]
    basefolder = os.path.join(basedir, cfg["basefolder_name"])
    trackpy_file = os.path.join(basefolder, cfg["trackpy_file_rel"])
    crop_params = cfg["crop_params"]
    ctracks_z_correction = cfg["ctracks_z_correction"]

    imageFolderBase = "recon_"
    ctracks_file = os.path.join(basefolder, r"ctracks_trajectories\linked_trajectories.csv")
    save_folder = os.path.join(basefolder, r"annotated_slices_new")

    crop_correction_ctracks = -1 * np.array([crop_params[2][0], crop_params[1][0], crop_params[0][0] + ctracks_z_correction])

    plot_trackpy = False
    plot_trackpy_only = False

    ctracks_df = pd.read_csv(ctracks_file)
    ctracks_df[['x', 'y', 'z']] += crop_correction_ctracks

    trackpy_df = None
    if plot_trackpy:
        trackpy_df = pd.read_csv(trackpy_file)
        trackpy_df.drop(index=0, inplace=True)
        save_folder += "_with_trackpy"

        if plot_trackpy_only:
            save_folder += "_only"
            ctracks_df = trackpy_df
            trackpy_df = None

    os.makedirs(save_folder, exist_ok=True)

    plot_slices(ctracks_df, recon_folder, imageFolderBase, crop_params, trackpy_df, save_folder)
    make_colorbar(os.path.join(save_folder, r"velocity_colorbar.png"))


if __name__ == "__main__":
    main()
