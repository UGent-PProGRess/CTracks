"""Piecewise-constant fit of per-voxel intensity dynamics across a time series of
reconstructed volumes, used to detect and localise dynamic changes (e.g. particle motion)
between successive reconstructions. Wraps a CUDA kernel (fit_steps.cu) that fits, per
voxel, a single step change between two intensity levels at some time t01.
"""

import os
import numpy as np
import cupy as cp
import tifffile

from ctrex.utils_gpu.base_kernel import CudaKernel, utils_gpu_path
from ctrex.utils import filetools as ft
from monstroCT.static_projectors import create_chunks

class VolumeSeries:
    """Lazily indexes a sequence of multi-page tiff volumes on disk (one file per
    timestep, matching `fmt`) and loads requested z-slices via memmap, so `fit_steps` can
    chunk through a dynamic-scan time series without holding everything in memory at once.
    Also reads intensity slope/offset/limits from the first volume's tiff description so
    results can be rescaled consistently.
    """
    def __init__(self, path, fmt, duration = None) -> None:
        self.path = path
        self.fmt = fmt
        self.tomo_names = ft.filelist(path, fmt)[:duration]
        len_prefix = fmt.find('%s')
        self.proj_indices = [int(index) for tomo_name in self.tomo_names
                             for index in os.path.basename(tomo_name)[len_prefix:].split('_')[:2]]
        self.descr, self.dtype = ft.read_description(filename=self.tomo_names[0])
        self.slope, self.offset = ft.split_description(self.descr)
        self.limits = ft.get_limits(self.descr, self.dtype)

    @property
    def duration(self):
        """Number of timesteps (volumes) in the series."""
        return len(self.tomo_names)

    def __len__(self):
        return len(self.tomo_names)

    def load_volume(self, ti, z_indices):
        """Loads the given z_indices slice for volume index ti without reading the whole
        tiff into memory (uses tifffile's `key=` selection).
        """
        with tifffile.TiffFile(self.tomo_names[ti]) as tif:
            series = tif.series[0]
            depth = len(series)
            z_indices = list(np.arange(0, depth)[z_indices])
            return series.asarray(key=z_indices)  # avoids full load


class VolumeFolderSeries(VolumeSeries):
    """Variant of VolumeSeries for a time series stored as one folder of single-slice
    tiffs per timestep (rather than one multi-page tiff per timestep); folder_fmt selects
    the per-timestep folders and fmt selects the slice files within each.

    NOTE: not currently exercised anywhere active - its only reference in the codebase is
    commented out (this file's own __main__ block).
    """
    def __init__(self, path, folder_fmt, fmt, duration = None) -> None:
        self.path = path
        self.folder_fmt = folder_fmt
        self.fmt = fmt
        self.tomo_names = ft.filelist(path, folder_fmt)
        self.tomo_names = [tomo_name for tomo_name in self.tomo_names if not tomo_name.endswith('.zip')]
        self.tomo_names = self.tomo_names[:duration]
        file_names = ft.filelist(self.tomo_names[0], fmt = fmt)
        self.descr, self.dtype = ft.read_description(filename=file_names[0])
        self.slope, self.offset = ft.split_description(self.descr)
        self.limits = ft.get_limits(self.descr, self.dtype)

    def load_volume(self, ti, z_indices):
        """Loads the requested z_indices as individual slice tiffs from the ti-th
        timestep folder.
        """
        file_names = np.asarray(ft.filelist(self.tomo_names[ti], fmt = self.fmt))[z_indices]  # np.asarray to support fancy indexing
        im, descr = ft.load_tiff(file_names[0])
        vol = np.zeros((len(file_names),) + im.shape, dtype = im.dtype)
        return ft.load(file_names, vol)


class FitStepsKernel(CudaKernel):
    """Thin wrapper that loads the fit_steps CUDA kernel (fit_steps.cu) via CudaKernel."""
    def __init__(self) -> None:
        super().__init__('fit_steps', os.path.join(utils_gpu_path, 'fit_steps.cu'))


def fit_steps(volume_series: VolumeSeries, thresh = 95, write_every = 5):
    """Fits a single step change (mu0 -> mu1 at time t01) per voxel across the volumes in
    volume_series, processing the volume in depth-chunks to bound GPU memory use, and
    periodically writes the current mu0/mu1/t01 maps out as tiffs (every write_every
    chunks, and always on the last chunk). Voxels whose |mu1-mu0| contrast is below the
    `thresh` percentile (computed over voxels seen so far) are reset to t01=0 to suppress
    noise.

    NOTE: not currently called from any active script - its only external reference
    (scripts/recon_multiphase.py) is commented out; it is still exercised via this file's
    own __main__ block.
    """
    fs = FitStepsKernel()

    first_rec = volume_series.load_volume(0, slice(None))
    mu0, mu1, t01 = [cp.zeros_like(first_rec, dtype = cp.float32) for _ in range(3)]

    # chunk
    depth, height, width = first_rec.shape
    chunks = create_chunks(depth, 0, 100)
    z_indices = np.arange(0, depth)
    for ci, chunk in enumerate(chunks):
        print(chunk)
        chunk_time = cp.zeros((volume_series.duration,) + first_rec[chunk.slice].shape, cp.float32)
        for ti in range(volume_series.duration):
            chunk_time[ti] = cp.asarray(volume_series.load_volume(ti, z_indices[chunk.slice]), dtype = cp.float32)

        chunk_time = (chunk_time * volume_series.slope + volume_series.offset).astype(cp.float32)

        fs(chunk_time[0].shape, chunk_time,
           mu0[chunk.slice], mu1[chunk.slice], t01[chunk.slice],
           cp.uint32(width), cp.uint32(height), cp.uint32(chunk.size), cp.uint32(volume_series.duration))

        if (ci % write_every != 0) and (ci != len(chunks) - 1):
            continue

        # filter out low contrast changes
        contrast = cp.abs(mu1 - mu0).get()
        t01[contrast < np.percentile(contrast[:chunk.offset + chunk.size], thresh)] = 0

        ft.write_tiff(os.path.join(volume_series.path, 'pwc_' + volume_series.fmt % 'mu0'), mu0.get(), volume_series.limits, np.uint16)
        ft.write_tiff(os.path.join(volume_series.path, 'pwc_' + volume_series.fmt % 'mu1'), mu1.get(), volume_series.limits, np.uint16)
        ft.write_tiff(os.path.join(volume_series.path, 'pwc_' + volume_series.fmt % 't01'), t01.get(), (0, volume_series.duration), np.uint16)


if __name__ == '__main__':
    vol_series = VolumeSeries(r'C:\Users\wannesg\Documents\working folder\16_12um_band2_mid_dyn_dra4\raw',
                                 'recon_time_%s.tif')
    # vol_series = VolumeFolderSeries(r'C:\Users\wannesg\Documents\working folder\16_12um_band2_mid_dyn_dra4\reconstructed Panthera',
    #                                 folder_fmt='recon_%s', fmt = '16_dra4_%s.tif', duration = None)
    fit_steps(vol_series)

