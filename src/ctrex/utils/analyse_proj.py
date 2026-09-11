"""Helpers for building reference/background projections and monitoring projection
intensity over time from a CTDataset sinogram, complementing the tiff-file-based versions
in filetools.py.
"""

import os
import numpy as np

from ctrex.utils import filetools as ft
from ctrex.utils.datasets import CTDataset
from ctrex.utils.filetools import printcounting


def create_reference_projections(scan_folder_full, scan_fmt, scan_folder_ref,
                                 io_file, di_file, proj_per_rot, num_rotations, height, width,
                                 ref_fun = np.max):
    """Builds one reference projection per rotation angle from a full scan directory
    (tiff files matched via scan_fmt), combining num_rotations repeated acquisitions with
    ref_fun (default max) after flat/dark correction.

    NOTE: not currently used/called anywhere in src/ or scripts/ - appears to be an
    earlier/duplicate version of the (currently used) create_reference_projections in
    src/ctracks/preprocessing/particle_subtraction.py.
    """
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


def create_reference_projections_dataset(ct_dataset: CTDataset, proj_per_rot, scan_folder_ref, scan_fmt, ref_fun = np.max):
    """Same idea as create_reference_projections, but reads directly from an in-memory
    CTDataset sinogram instead of tiff files on disk, undoing the log-transform before
    combining.

    NOTE: not currently called anywhere - its only reference in the codebase is
    commented out (scripts/recon_multiphase.py).
    """
    os.makedirs(scan_folder_ref, exist_ok=True)
    limits, dtype = (0, 1.4), np.uint16
    print(proj_per_rot)
    for angle in range(0, int(proj_per_rot), 1):
        printcounting(str(angle), angle, int(proj_per_rot))
        # create float range first with float proj_per_rot, then convert to integer to retain final precision
        same_angle_range = np.arange(angle, ct_dataset.num_projections, float(proj_per_rot))
        same_angle_range = same_angle_range.astype(np.int32)
        projections = (- ct_dataset[:,same_angle_range,:].detach()).exp().cpu().numpy()  # convert back to normalised
        ref_projection = ref_fun(projections, axis = 1)
        ft.write_tiff(os.path.join(scan_folder_ref, scan_fmt % f"{angle:06d}"), ref_projection, limits, dtype)

        # if angle == 0:
        #     # print(same_angle_range)
        #     ft.write_tiff(os.path.join(scan_folder_ref, "_start_angle_" + scan_fmt % ""), np.swapaxes(projections, 0,1),
        #                   limits, dtype)


def monitor_intensities_time(ct_dataset, proj_per_rot):
    """Computes the mean (post log-transform-undone) intensity per full rotation across
    a scan, one value per rotation, for tracking how average projection intensity changes
    over time.

    NOTE: not currently called anywhere - its only reference in the codebase is
    commented out (scripts/recon_multiphase.py).
    """
    num_rotations = int(ct_dataset.num_projections / proj_per_rot)
    intensities = np.zeros((num_rotations,), dtype = np.float32)
    for rot in range(0, num_rotations):
        printcounting(f"{rot} / {num_rotations}", rot, num_rotations)
        projections = (- ct_dataset[:,int(rot * proj_per_rot): int((rot + 1) * proj_per_rot),:].detach()).exp().cpu().numpy()
        intensity = np.mean(projections)
        intensities[rot] = intensity
    return intensities
