"""Low-level geometry/data helpers backing `PoreStructure.py` and `Particle.py`.

--------------------------------------------------------------------------
Role in the data flow
--------------------------------------------------------------------------
- `tif_2_geo` + `geo_2_mesh`: the two steps of mesh generation used by
  `PoreStructure_CT.get_surface()` - marching cubes on a segmented volume to get raw
  (vertices, faces), then converting that into a smoothed pyvista mesh.
- `convert_np_to_df`: an alternate particle-iteration data source used by
  `ParticleIterator_CTrack` (see `Particle.py`) - converts raw `scan_{i}.npy`
  reconstruction files into the same (frame, x, y, z, vx, vy, vz) CSV shape that
  `ParticleIterator_DF` reads directly. NOTE: this independently reimplements the
  same scan-file -> position/velocity extraction logic as
  `analysis_functions.py::load_ctrack_files` in the parent analysis package; kept
  separate intentionally (a prior cleanup pass decided de-duplicating it would mean
  changing which module `vis3D` code depends on) rather than deduplicated here.
"""
import os

import numpy as np
import pandas as pd
import pyvista as pv

from skimage import measure


def tif_2_geo(tif_file, threshold=0, down_sample_factor=4):
    """
    Extract the surface geometry (vertices, faces) of the `threshold`-valued
    voxels in `tif_file` via marching cubes, after downsampling by
    `down_sample_factor`. Vertex coordinates are scaled back up by
    `down_sample_factor` so they stay in the original volume's coordinate
    frame.
    """
    img = tif_file == threshold
    img = img[::down_sample_factor, ::down_sample_factor, ::down_sample_factor]
    verts, faces, _, _ = measure.marching_cubes(img, level=0.5)
    verts = verts * down_sample_factor
    return verts, faces


def geo_2_mesh(verts, faces, smooth_iter=10, smooth_factor=0.5):
    """
    Convert marching-cubes geometry (vertices, triangle faces) into a
    smoothed pyvista PolyData mesh.

    NOTE: `smooth_iter`/`smooth_factor` are accepted but currently NOT used -
    the smoothing call below is hardcoded to n_iter=10, relaxation_factor=0.5
    (which happen to match the defaults). Left as-is since fixing this would
    change output for any caller passing non-default values; flagged here
    for whoever touches this next.
    """
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    faces_pv = faces_pv.flatten()
    mesh = pv.PolyData(var_inp=verts, faces=faces_pv)
    mesh = mesh.smooth(n_iter=smooth_iter, relaxation_factor=smooth_factor)
    return mesh

def convert_np_to_df(frame_files, track_set, filename = "ctracks_df.csv"):
    """Build and save a per-particle position/velocity DataFrame from a list
    of `scan_{i}.npy` reconstruction result files.

    For each file (one per rotation/frame), loads the pickled results dict,
    takes the `track_set` entry (e.g. "reconstruction" or "ground truth"),
    and derives per-particle position (mean over the track) and velocity
    (last - first point of the track) from its (N, T, 3) position array.
    Writes the combined result as a CSV (columns: frame, x, y, z, vx, vy, vz,
    velMag) next to the first input file, named `filename`, and returns that
    path.

    NOTE: this is an independent reimplementation of the same
    scan_{i}.npy -> position/velocity extraction logic as
    `analysis_functions.py::load_ctrack_files` in the parent analysis
    package - kept separate intentionally (this module is used when plotting
    directly from raw recon .npy files rather than a pre-linked CSV), not
    deduplicated here.
    """
    ctrack_pos, ctrack_vel, ctrack_mag = [], [], []

    for file in frame_files:
        ctrack_results = np.load(file, allow_pickle=True).item()
        ctrack_recon = ctrack_results[track_set][0]
        # TODO change this to use the actual track/gradient
        pos = ctrack_recon.mean(axis=1)
        vel = ctrack_recon[:, -1, :] - ctrack_recon[:, 0, :]
        mag = np.linalg.norm(vel, axis=1, keepdims=True)

        ctrack_pos.append(pos)
        ctrack_vel.append(vel)
        ctrack_mag.append(mag)

    # Calculate counts for the frame column
    counts = [t.shape[0] for t in ctrack_pos]
    frames = np.repeat(np.arange(len(frame_files)), counts)

    # Each frame_data will be (N, 7) -> [x, y, z, vx, vy, vz, velMag]
    all_features = [np.hstack([p, v, m]) for p, v, m in zip(ctrack_pos, ctrack_vel, ctrack_mag)]
    combined_data = np.vstack(all_features)

    # Build the final DataFrame
    cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'velMag']
    df = pd.DataFrame(combined_data, columns=cols)
    df.insert(0, 'frame', frames)

    first_file_path = frame_files[0]
    folder_path = os.path.dirname(first_file_path)
    save_path = os.path.join(folder_path, filename)

    df.to_csv(save_path, index=False)
    print(f"DataFrame saved to: {save_path}")
    return save_path
