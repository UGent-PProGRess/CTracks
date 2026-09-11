"""Slice a reconstructed TIFF-stack volume series along the three axes and
save the resulting orthogonal slice images (one set per time frame) to disk.

Standalone entry-point script - not imported elsewhere. Select the dataset
to process via ACTIVE_DATASET below.
"""

import collections.abc
collections.Iterable = collections.abc.Iterable  # pims compatibility shim (expects the pre-3.10 collections.Iterable alias)
import pims
import tifffile as tf
import os
from scripts.particle_tracking.data_analysis.vis2D import stackReader as sr

# Data root - change this if CTracksSubmissionData is moved or mounted under a
# different drive/path (e.g. moving to a new PC).
DATA_ROOT = "J:\\"
_CTSD = os.path.join(DATA_ROOT, r"CTracksSubmissionData\Raw data\Experimental")

# Dataset configs: source recon folder, output folder, and the fixed
# yz/xz/xy display-slice coordinates for that sample geometry.
DATASETS = {
    "porous_180nlmin": dict(
        basefolder=os.path.join(_CTSD, r"Porous glass\180nlmin\recon"),
        imageDirExtension="recon_",
        outfolder=os.path.join(_CTSD, r"Porous glass\Reslices\180nlmin"),
        xDisplay=330, yDisplay=330, zDisplay=474,
    ),
    "porous_60nlmin": dict(
        basefolder=os.path.join(_CTSD, r"Porous glass\60nlmin\recon"),
        imageDirExtension="recon_",
        outfolder=os.path.join(_CTSD, r"Porous glass\Reslices\60nlmin"),
        xDisplay=330, yDisplay=330, zDisplay=474,
    ),
    "porous_1ulmin": dict(
        basefolder=os.path.join(_CTSD, r"Porous glass\1ulmin\recon"),
        imageDirExtension="recon_",
        outfolder=os.path.join(_CTSD, r"Porous glass\Reslices\1ulmin"),
        xDisplay=330, yDisplay=330, zDisplay=474,
    ),
    "capillary_60nlmin": dict(
        basefolder=os.path.join(_CTSD, r"Capillary\60nlmin\recon"),
        imageDirExtension="recon_",
        outfolder=os.path.join(_CTSD, r"Capillary\Reslices\60nlmin"),
        xDisplay=180, yDisplay=180, zDisplay=474,
    ),
    "capillary_180nlmin": dict(
        basefolder=os.path.join(_CTSD, r"Capillary\180nlmin\recon"),
        imageDirExtension="recon_",
        outfolder=os.path.join(_CTSD, r"Capillary\Reslices\180nlmin"),
        xDisplay=180, yDisplay=180, zDisplay=474,
    ),
    "capillary_1ulmin": dict(
        basefolder=os.path.join(_CTSD, r"Capillary\1ulmin\recon"),
        imageDirExtension="recon_",
        outfolder=os.path.join(_CTSD, r"Capillary\Reslices2\1ulmin"),
        xDisplay=180, yDisplay=180, zDisplay=474,
    ),
}
ACTIVE_DATASET = "capillary_1ulmin"

# Volume crop margins applied before slicing, as ((z0,z1),(y0,y1),(x0,x1)).
CROP_PARAMS_OPTIONS = {
    "none": ((0, 0), (0, 0), (0, 0)),
    "150": ((150, 150), (150, 150), (150, 150)),
}
ACTIVE_CROP_PARAMS = "none"


def main():
    """Load the active dataset's reconstruction stack, crop it, and save the
    yz/xz/xy slice through (xDisplay, yDisplay, zDisplay) for every frame.
    """
    cfg = DATASETS[ACTIVE_DATASET]
    basefolder = cfg["basefolder"]
    imageDirExtension = cfg["imageDirExtension"]
    outfolder = cfg["outfolder"]
    xDisplay = cfg["xDisplay"]
    yDisplay = cfg["yDisplay"]
    zDisplay = cfg["zDisplay"]
    cropParams = CROP_PARAMS_OPTIONS[ACTIVE_CROP_PARAMS]

    print("Loading frames")
    frames = sr.set_pipeline(sr.stackReader(basefolder, imageDirExtension))
    frames.bundle_axes = ['z', 'y', 'x']
    frames.iter_axes = 't'
    frames = pims.process.crop(frames, cropParams)

    print("Saving slices")

    slices_axes = [(slice(None), slice(None), xDisplay), (slice(None), yDisplay, slice(None)), (zDisplay, slice(None), slice(None))]
    out_directories = [os.path.join(outfolder, f"yz_x{xDisplay}"), os.path.join(outfolder, f"xz_y{yDisplay}"), os.path.join(outfolder, f"xy_z{zDisplay}")]

    for dir in out_directories:
        os.makedirs(dir, exist_ok=True)

    for i in range(len(frames)):
        image = frames[i]
        for j in range(len(out_directories)):
            tf.imwrite(os.path.join(out_directories[j], f"frame{i}.tif"), image[slices_axes[j]])


if __name__ == "__main__":
    main()
