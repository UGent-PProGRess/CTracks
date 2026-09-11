import os
import ctrex.utils.filetools as ft
import numpy as np
from skimage.filters import threshold_yen, median  # find imageJ's max entropy threshold in skimage
from skimage.morphology import disk


def create_reference_projections(scan_folder_full, scan_fmt, scan_folder_ref,
                                 io_file, di_file, proj_per_rot, num_rotations, height, width,
                                 ref_fun = np.max):
    """Build a flat/dark-corrected reference projection set from a full multi-rotation scan by
    reducing (via ``ref_fun``, e.g. max or mean) all rotations at each projection angle into one
    reference projection, and writing the result to ``scan_folder_ref``. Used to construct a
    per-angle background/reference to later subtract from a difference scan."""
    os.makedirs(scan_folder_ref, exist_ok=True)
    filelist = ft.filelist(scan_folder_full, fmt=scan_fmt)
    flat, _ = ft.load_tiff(io_file)
    dark, _ = ft.load_tiff(di_file)
    for i in range(0, proj_per_rot, 1):
        print(i)
        projections = ft.load(filelist[i::proj_per_rot], np.zeros((num_rotations, height, width), flat.dtype),
                              None, False)
        projections = projections.astype(np.float32)
        projections = (projections - dark) / (flat - dark)

        ref_projection = ref_fun(projections, axis = 0)
        ft.write_tiff(os.path.join(scan_folder_ref, scan_fmt % f"{i:06d}"),
                      ref_projection, [0, 1.4], np.uint16)


def find_particles(recon_volume, median_radius = 8, threshold_function = threshold_yen):
    """Detect particles as the positive residual between the reconstructed volume and its
    per-slice median filter (particles stand out as locally brighter than their surroundings),
    thresholded (via ``threshold_function``) on the central slice."""
    footprint = disk(median_radius)
    recon_volume_median = recon_volume.copy()
    for si, slc in enumerate(recon_volume_median):
        recon_volume_median[si] = median(recon_volume[si], footprint=footprint)
    recon_volume_diff = recon_volume - recon_volume_median
    half_height = recon_volume.shape[0] // 2
    particles_mask = recon_volume_diff > threshold_function(recon_volume_diff[half_height])
    return particles_mask


def subtract_particles(recon_volume, particles_mask, fluid_mean, fluid_std):
    """Zero out particle voxels (per ``particles_mask``) and replace them with samples drawn from
    a Normal(``fluid_mean``, ``fluid_std``) distribution, approximating the surrounding fluid."""
    recon_volume_no_particles = recon_volume * (particles_mask == 0)
    nonzero = np.nonzero(particles_mask)
    replacement_values = np.random.normal(loc=fluid_mean, scale=fluid_std, size=len(nonzero[0]))
    recon_volume_no_particles[*nonzero] = replacement_values
    return recon_volume_no_particles


def find_subtract_particles(volume_file, mask_file = None, fluid_mean = 0., fluid_std = 0.):
    """Load a reconstructed volume and produce a particle-free version: detects particles
    (``find_particles``) unless a precomputed ``mask_file`` is given, then subtracts them
    (``subtract_particles``). Returns (volume_no_particles, particles_mask, tiff scaling limits)."""
    recon_volume, limits = ft.load_and_scale_tiff(volume_file)
    if mask_file is None:
        particles_mask = find_particles(recon_volume)
    else:
        particles_mask = ft.load_tiff(mask_file)[0]

    recon_volume_no_particles = subtract_particles(recon_volume, particles_mask, fluid_mean, fluid_std)
    return recon_volume_no_particles, particles_mask, limits


# Standalone entry-point config: which reconstructed volume to strip particles from.
MAIN_CONFIG = dict(
    scan_folder = r'C:\Users\wannesg\Documents\working folder\glass_HQ',
    scan_name = 'recon_glass_HQ_0248_1648_b2_0',
    fluid_mean = 0.87,
    fluid_std = 0.022,
)


def main():
    cfg = MAIN_CONFIG
    scan_folder = cfg["scan_folder"]
    scan_name = cfg["scan_name"]
    recon_volume_file = os.path.join(scan_folder, scan_name + '.tif')
    recon_mask_file = os.path.join(scan_folder, scan_name + '_particles.tif')
    recon_volume_file_no_particles = os.path.join(scan_folder, scan_name + '_no_particles.tif')
    recon_volume_no_particles, particles_mask, limits = find_subtract_particles(
        recon_volume_file, recon_mask_file, fluid_mean = cfg["fluid_mean"], fluid_std = cfg["fluid_std"])
    ft.write_tiff(recon_volume_file_no_particles, recon_volume_no_particles, limits, dtype = np.uint16)


if __name__ == '__main__':
    main()
