#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:00:06 2022

@author: ext-bultreys_t
"""

import os
import pims
from pims import FramesSequenceND, Frame
from skimage.io import MultiImage
import numpy as np
import psutil

import re


@pims.pipeline
def set_pipeline(image):
    """Identity pims pipeline step (wraps `image` unchanged) so a `stackReader` can be
    composed with other `pims.pipeline`/`pims.process` steps (e.g. `pims.process.crop`).

    Shared by `plot_slices.py` and `plot_slices_tracks.py`, which both build a
    `stackReader` frame sequence this way before slicing/cropping it.
    """
    return image


def get_last_number_from_path(path):
    """Return the last integer found in `path`, or 0 if none is found.

    Used to sort per-frame reconstruction subfolders numerically (e.g.
    "recon_2" before "recon_10") instead of lexicographically.
    """
    numbers = re.findall(r'\d+', path)
    if numbers:
        return int(numbers[-1])
    return 0


class stackReader(FramesSequenceND):
    """A `pims.FramesSequenceND` over a folder of per-frame TIFF-stack subfolders.

    Each subfolder under `superfolder` whose name contains `imageFolderBase`
    (e.g. "recon_0", "recon_1", ...) is treated as one time frame, and is
    itself a stack of `fileExtension` images along the z-axis. Frames are
    lazily loaded on indexing via `get_frame`.
    """

    def __init__(self, superfolder="", imageFolderBase="", fileExtension="tif"):
        super(stackReader, self).__init__()
        self.superfolder = superfolder
        self.imageFolderBase = imageFolderBase
        self.folderList = [f.path for f in
                           sorted(os.scandir(superfolder), key=lambda e: get_last_number_from_path(e.name)) if
                           (f.is_dir() and imageFolderBase in f.name)]
        self.fileExtension = fileExtension

        firstImage = MultiImage(os.path.join(self.folderList[0], "*." + self.fileExtension))
        self._init_axis('z', len(firstImage))
        self._init_axis('y', firstImage[0].shape[1])
        self._init_axis('x', firstImage[0].shape[0])
        self._init_axis('t', len(self.folderList))
        self._len = len(self.folderList)
        self._dtype = firstImage[0].dtype
        self._frame_shape = (len(firstImage), firstImage[0].shape[1], firstImage[0].shape[0])

    def get_frame(self, t):
        """Load and return frame `t` (a time index into `folderList`) as a `pims.Frame`."""
        multiImage = MultiImage(os.path.join(self.folderList[t], "*." + self.fileExtension))
        print('RAM memory % used:', psutil.virtual_memory()[2])
        stack = Frame(np.stack(multiImage, axis=0), frame_no=t)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        return stack

    def __len__(self):
        return self._len

    @property
    def frame_shape(self):
        return self._frame_shape

    @property
    def pixel_type(self):
        return self._dtype
