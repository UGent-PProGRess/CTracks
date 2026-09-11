"""
Build a particle-free reference scan from a dynamic scan, by taking (per default) the
max-intensity projection across all rotations at each viewing angle (0 deg, 360 deg,
720 deg, ...; 1 deg, 361 deg, 721 deg, ...; etc). This reference scan is later
subtracted from the dynamic scan (see the recon_differences_* scripts) to isolate the
particles.
"""

import os

import numpy as np

from ctrex.utils import filetools as ft

# how to combine same-angle projections across rotations into the reference: max, min or median
REFERENCE_TYPE = np.max

PROJ_PER_ROT = 850

# ---------------------------------------------------------------------------
# Data root - change this if CTracksSubmissionData is moved or mounted under a
# different drive/path (e.g. moving to a new PC).
# ---------------------------------------------------------------------------
DATA_ROOT = "J:\\"

# ---------------------------------------------------------------------------
# Per-dataset scan configuration - select the active one below.
# ---------------------------------------------------------------------------
DATASETS = {
    "60nlmin_porous": dict(
        recon_name="recon_60nlmin",
        scan_fmt='scan_%s.tif',
        scan_folder_full=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Porous glass\60nlmin"),
        total_scan_rotations=90,
        scan_folder_base=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Porous glass\60nlmin"),
    ),
    "180nlmin_porous": dict(
        recon_name="recon_180nlmin",
        scan_fmt='180nlmin_%s.tif',
        scan_folder_full=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Porous glass\180nlmin"),
        total_scan_rotations=60,
        scan_folder_base=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Porous glass\180nlmin"),
    ),
    "1ulmin_porous": dict(
        recon_name="recon_1ulmin",
        scan_fmt='1ulmin_%s.tif',
        scan_folder_full=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Porous glass\1ulmin"),
        total_scan_rotations=20,
        scan_folder_base=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Porous glass\1ulmin"),
    ),
    "60nlmin_capillary": dict(
        recon_name="recon_60nlmin",
        scan_fmt='60nlmin_%s.tif',
        scan_folder_full=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Capillary\60nlmin"),
        total_scan_rotations=60,
        scan_folder_base=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Capillary\60nlmin"),
    ),
    "180nlmin_capillary": dict(
        recon_name="recon_180nlmin",
        scan_fmt='180nlmin_%s.tif',
        scan_folder_full=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Capillary\180nlmin"),
        total_scan_rotations=60,
        scan_folder_base=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Capillary\180nlmin"),
    ),
    "1ulmin_capillary": dict(
        recon_name="recon_1ulmin",
        scan_fmt='1ulmin_%s.tif',
        scan_folder_full=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental\Capillary\1ulmin"),
        total_scan_rotations=40,
        scan_folder_base=os.path.join(DATA_ROOT, r"CTracksSubmissionData\Processed results\Experimental\Capillary\1ulmin"),
    ),
}
ACTIVE_DATASET = "180nlmin_porous"
cfg = DATASETS[ACTIVE_DATASET]
recon_name = cfg["recon_name"]
scan_fmt = cfg["scan_fmt"]
scan_folder_full = cfg["scan_folder_full"]
total_scan_rotations = cfg["total_scan_rotations"]
scan_folder_base = cfg["scan_folder_base"]
scan_folder_ref = os.path.join(scan_folder_base, "background_projections2")


def main():
    """Generate the particle-free reference scan for the active dataset and write it
    to <scan_folder_base>/<max|min|median>_intensity_normalised."""
    io_file = os.path.join(scan_folder_full, "io000001.tif")
    di_file = os.path.join(scan_folder_full, "di000001.tif")

    # create reference projections that filter out particles by taking the max intensity projection
    # over the projections at same viewing angles (0°, 360°, ...), (1°, 361°, ... ), etc.
    ft.create_reference_scan(scan_folder_full, scan_folder_ref, scan_fmt, io_file, di_file, PROJ_PER_ROT, total_scan_rotations,
                             start_rotation=0, ref_func=REFERENCE_TYPE)
    print("Reference scan generated.")


if __name__ == "__main__":
    main()
