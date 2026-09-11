"""Sinogram dataset hierarchy: torch ``Dataset`` classes that give indexed (v, projection, u)
access to a scan's raw/projection data, regardless of where that data actually lives.

``Detector`` is the plain shape descriptor (detector height/width, number of projections, ROI)
shared by all of them, with no data or acquisition geometry attached - that lives in
``ctrex.utils.trajectories``. ``CTDataset`` is the common base, implementing shared index
parsing and defining the ``__getitem__``/``__setitem__`` contract.

The subclasses trade off where the sinogram data is stored and how much of it is kept in
memory at once:
- ``CTDatasetInMemory`` holds the full sinogram tensor in memory (or on the given device).
- ``CTDatasetSparse`` keeps only a fixed-size rolling buffer of projections in memory, with
  zero-fill for anything not currently buffered.
- ``CTDatasetOnDisk`` (a ``CTDatasetSparse``) lazily loads projections from a folder of
  per-projection TIFF files, using the sparse buffer as an in-memory cache and applying
  flat/dark-field correction to produce optical-depth sinograms.
- ``CTDatasetOnFile`` (a ``CTDatasetOnDisk``) is the same idea but for a single multi-page
  TIFF file instead of one file per projection.
- ``CTDatasetOnH5File`` (a ``CTDatasetOnFile``) reads from a single HDF5 dataset instead.
- ``CTDatasetMogno`` (a ``CTDatasetOnH5File``) additionally remaps between the physical,
  gapped, tiled detector layout of the "Mogno" beamline and a contiguous framed grid.

``CTDatasetDifference`` and ``CTDatasetAverage`` are decorator-style datasets that wrap one
or two other ``CTDataset`` instances rather than sourcing data themselves: the former
returns/writes the difference between a dynamic and a (wrapped) reference dataset, the
latter averages consecutive projections of an underlying dataset together.
"""

import os
import numpy as np
import warnings
from typing import Union
import torch
from torch.utils.data import Dataset
import tifffile
import h5py

from ctrex.utils import filetools as ft


# Todo: check https://napari.org/stable/tutorials/processing/dask.html


class Detector:  # Data shape descriptor, no data or geometry here
    """Plain descriptor of a detector's data shape: height, width, number of projections, and
    an optional region of interest (ROI). Carries no pixel data and no acquisition geometry
    (angles, source/detector positions, etc.) - those live in ``ctrex.utils.trajectories``.
    ``CTDataset`` (and everything in this module) inherits from this to get shape/ROI handling
    for free alongside its actual data-loading behaviour.
    """
    def __init__(self, height: int = 1, width: int = 1, num_projections: int = 0, roi = None, **kwargs):
        self.width = width
        self.height = height
        self.roi = roi
        self.num_projections = num_projections

    @property
    def num_projections_each(self):
        return [self.num_projections]

    @property
    def roi(self):
        return self._roi if self._roi is not None else (slice(0, self.height), slice(0, self.width))
    
    @roi.setter
    def roi(self, newroi):
        """Set the ROI, clamping the given (v, u) slices to stay within (height, width).
        ``None`` clears the ROI, so ``roi``/``rheight``/``rwidth`` fall back to the full frame."""
        if newroi is not None:
            self._roi = (slice(max(0, newroi[0].start), min(self.height, newroi[0].stop)),
                         slice(max(0, newroi[1].start), min(self.width, newroi[1].stop)))
        else:
            self._roi = None

    @property
    def rheight(self) -> int:
        return self.roi[0].stop - self.roi[0].start
    
    @property
    def rwidth(self) -> int:
        return self.roi[1].stop - self.roi[1].start

    @property
    def dimension(self) -> int:
        return 2 if self.height == 1 else 3

    @property
    def sino_params(self):
        sino_params = {'width': self.width, 'height': self.height, 'dimension': self.dimension}
        return sino_params


class CTDataset(Dataset, Detector):
    """A dataset that defaults to sampling along the second dimension."""
    def __init__(self, sino_params, projection_times: torch.Tensor):
        Detector.__init__(self, **sino_params)
        self.projection_times = projection_times
        self.scan_folder = '.'
        self.scan_name = 'scan_data'
            
    @property
    def shape(self):
        return self.height, self.num_projections, self.width

    @property
    def device(self):
        return self.projection_times.device

    @device.setter
    def device(self, device):
        self.projection_times = self.projection_times.to(device)
    
    def _parse_indices(self, batch_indices):
        """Normalize whatever indexing was passed to ``__getitem__``/``__setitem__`` into an
        explicit ``(v_indices, projection_indices, u_indices)`` triple. Accepts a bare
        projection index/slice/tensor (v and u default to the full ROI), or a 1-3 element
        tuple giving some/all of (v, projection, u). ``projection_indices`` is always
        returned as an int tensor on ``self.device``; ``v_indices``/``u_indices`` are passed
        through as-is (slice, tensor, etc.)."""
        pass
        if isinstance(batch_indices, tuple):
            if len(batch_indices) == 1:
                v_indices, projection_indices, u_indices = slice(None), batch_indices, slice(None)
            elif len(batch_indices) == 2:
                v_indices, projection_indices, u_indices = batch_indices[0], batch_indices[1], slice(None)
            else:  # 3
                v_indices, projection_indices, u_indices = batch_indices
        else:
            v_indices, projection_indices, u_indices = slice(None), batch_indices, slice(None)
        # Convert potential slice(None) to ensure tensor format
        # v_indices = torch.arange(0, self.height, device = self.device)[v_indices]
        if isinstance(projection_indices, torch.Tensor):
            projection_indices = projection_indices
        elif isinstance(projection_indices, slice):
            start,stop,step = projection_indices.indices(self.num_projections)
            projection_indices = torch.as_tensor(list(range(start, stop, step)))
        else:
            projection_indices = torch.as_tensor(projection_indices)
        projection_indices = torch.atleast_1d(projection_indices).to(self.device).to(torch.int)
        # u_indices = torch.arange(0, self.width, device = self.device)[u_indices]
        return v_indices, projection_indices, u_indices

    def select_indices(self, select):
        """ https://discuss.pytorch.org/t/intersection-between-to-vectors-tensors/50364/9 """
        if isinstance(select, slice):
            intersection = self.projection_times[select]
            return intersection
        a_cat_b, counts = torch.cat([self.projection_times, select]).unique(return_counts=True)
        intersection = a_cat_b[torch.where(counts.gt(1))].cpu().numpy()
        return intersection

    def __getitem__(self, vpu_indices):
        """Abstract - the base implementation just forwards to ``Dataset.__getitem__``, which
        raises ``NotImplementedError``. Every concrete subclass overrides this to actually
        return the requested sinogram data."""
        super().__getitem__(vpu_indices)

    def __setitem__(self, vpu_indices, sinogram_slices):
        raise NotImplementedError


class CTDatasetInMemory(CTDataset):
    """Simplest ``CTDataset``: wraps a single, already-allocated ``sinograms`` tensor
    (shape ``(height, num_projections, width)``) held fully in memory (or on whichever
    device it's on), with no lazy loading or caching involved. Used for ground-truth/
    simulated data and as the reconstruction buffer for simulated volumes, e.g. via
    ``CTModule.empty_sinogram``.
    """
    def __init__(self, sino_params, projection_times, sinograms, roi = None):
        """
        projection_times: (num_projections,)
        ground_truth: (detector_height, num_projections, detector_width)
        """
        _sino_params = {'width': sinograms.shape[2], 'height': sinograms.shape[0],
                        'num_projections': sinograms.shape[1], 'roi': roi}
        _sino_params.update(sino_params)
        self.sinograms = sinograms
        super().__init__(_sino_params, projection_times)

    @property
    def device(self):
        return self.sinograms.device
    
    @device.setter
    def device(self, device):
        if self.sinograms.device != device:
            warnings.warn("device of sinograms changed to correspond to projection times tensor device")
            self.sinograms = self.sinograms.to(device)

    def __getitem__(self, vpu_indices):
        """
        Returns sampled ground_truth based on batch indices.
        """
        # batch_indices = torch.tensor(batch_indices, device = self.sinogram.device)  # Ensure tensor format
        # batch_coordinates = self.projection_times[batch_indices]  # Shape: (batch_size)
        vpu_indices = self._parse_indices(vpu_indices)
        batch_sinograms = self.sinograms[vpu_indices]  # Shape: (detector_height, batch_size, detector_width)
        return batch_sinograms
        # return batch_coordinates, batch_sinograms, batch_indices
    
    def __setitem__(self, vpu_indices, sinogram_slices):
        """Write ``sinogram_slices`` directly into the in-memory ``sinograms`` tensor at the
        given (v, projection, u) indices."""
        vpu_indices = self._parse_indices(vpu_indices)
        self.sinograms[vpu_indices] = sinogram_slices


class CTDatasetSparse(CTDataset):
    """Unlike ``CTDatasetInMemory``, does not hold the full sinogram: only a fixed-size
    rolling buffer of ``mem_projections`` projections is kept in memory at once
    (``mem_sinograms``/``mem_indices``), and any projection not currently buffered reads
    back as zeros. New writes first fill open buffer slots, then overwrite the oldest
    non-``fixed_memory`` slots once the buffer is full (see ``__setitem__``). This is the
    base for the disk/file-backed datasets below, which use the buffer as an in-memory
    cache over slower storage; used directly it also serves as a fixed-memory reconstruction
    buffer (e.g. via ``CTModule.empty_sinogram``).
    """
    def __init__(self, sino_params, projection_times, device, mem_projections, roi = None,
                 fixed_memory = 0):
        """
            Stores only filled projections in memory, with zero-fill on retrieval.
        """
        self.projection_times = projection_times.to(device)
        self.num_projections = int(self.projection_times[-1] - self.projection_times[0] + 1)  # possible gaps
        self.device = device
        # Track which projections are filled
        self.mem_projections = max(1, min(mem_projections, self.num_projections))
        self.fixed_memory = fixed_memory
        sino_params['num_projections'] = self.num_projections
        sino_params['roi'] = roi
        Detector.__init__(self, **sino_params)

        self._init_memory()
        
    def _init_memory(self):
        """(Re-)allocate the empty rolling buffer: the filled-flag mask, the buffer slot ->
        projection index mapping (all slots start "empty" as -1), and the zero-filled
        sinogram buffer itself. Called on construction and whenever the ROI changes."""
        self.all_stored = torch.zeros(self.num_projections, dtype = torch.bool, device = self.device)
        # "empty" = -1
        self.mem_indices = -1 * torch.ones(self.mem_projections, dtype=torch.long, device = self.device)
        self.mem_sinograms = torch.zeros(self.rheight, self.mem_projections, self.rwidth,
                                         dtype = torch.float32, device = self.device)
    
    @property
    def roi(self):
        return super().roi
    
    @roi.setter
    def roi(self, newroi):
        """Set the ROI and reset the buffer via ``_init_memory`` (the buffer's spatial shape
        depends on the ROI, and any previously cached data is discarded)."""
        CTDataset.roi.fset(self, newroi)
        # super(CTDatasetSparse, self.__class__).roi.fset(self, newroi)
        self._init_memory()
    
    def __setitem__(self, indices, sinogram_slices):
        """
        Set multiple slices in memory-efficient form.
        
        Args:
            indices: list or tensor of projection indices
            sinogram_slices: (H, B, W) tensor where B = len(indices)
        """
        vpu_indices = self._parse_indices(indices)
        projection_indices = vpu_indices[1]
        assert len(projection_indices) <= self.mem_projections, 'not enough memory for this update'
        projection_indices = torch.as_tensor(projection_indices, dtype = self.mem_indices.dtype, device=self.device)
        if sinogram_slices.shape != (self.rheight, len(projection_indices), self.rwidth):
            return
        sinogram_slices = sinogram_slices.to(self.device)
        
        # check which projections are already stored and where
        keys_found, keys_missing, buffer_indices = self.get_filled_indices(projection_indices)

        # overwrite existing projections
        assert (self.mem_indices[buffer_indices] == projection_indices[keys_found]).all()
        self.mem_sinograms[:,buffer_indices] = sinogram_slices[:,keys_found]
        
        # first fill buffer starting from previous open spot
        open_spots = 1*(self.mem_indices < 0)
        if torch.any(open_spots):
            first_open = torch.argmax(open_spots)  # -1 is not filled
            remainder_1 = keys_missing[:len(self.mem_indices) - first_open]
            self.mem_indices[first_open:first_open + len(remainder_1)] = projection_indices[remainder_1]
            self.mem_sinograms[:,first_open:first_open + len(remainder_1)] = sinogram_slices[:, :len(remainder_1)]
            # self.all_stored[self.mem_indices[first_open:first_open + len(remainder_1)]] = True
            self.all_stored[projection_indices[remainder_1]] = True
        else:
            remainder_1 = keys_missing[:0]
            first_open = len(self.mem_indices)
        
        # overwrite previous projections that were not stored already
        remainder_2 = keys_missing[len(remainder_1):]
        if len(remainder_2) > 0:
            open_spots[:] = 1
            open_spots[buffer_indices] = 0  # these are used
            open_spots[:self.fixed_memory] = 0  # these should not be overwritten
            open_spot_indices = torch.nonzero(open_spots)[:len(remainder_2),0]
            self.all_stored[self.mem_indices[open_spot_indices]] = False
            self.mem_indices[open_spot_indices] = projection_indices[remainder_2]
            self.mem_sinograms[:,open_spot_indices] = sinogram_slices[:,remainder_2]  # like remainder_2  # previously: len(keys_missing)-len(remainder_2):
            # self.all_stored[self.mem_indices[open_spot_indices[:len(remainder_2)]]] = True
            self.all_stored[self.mem_indices[open_spot_indices]] = True
        num_stored = self.all_stored.sum()  # if suddenly lower than mem_projections, there are duplicates
        pass
    
    def __getitem__(self, batch_indices):
        """
        Retrieve a dense (zero-filled) view over requested projection indices.
        
        Args:
            batch_indices: list or tensor of projection indices
        Returns:
            batch_sinograms: (H, buffer, W)
        """
        vpu_indices = self._parse_indices(batch_indices)

        v_indices, projection_indices, u_indices = vpu_indices
        # batch_coordinates = self.projection_times[batch_indices]  # Shape: (batch_size)

        # TODO if slicing spatial dimensions, still returns sinogram with shape of ROI
        try:
            sheight = len(np.arange(self.rheight)[v_indices])  # selected height - slice(None) selects roi
        except TypeError:
            sheight = 1
        try:
            swidth = len(np.arange(self.rwidth)[u_indices])  # selected width
        except TypeError:
            swidth = 1
        batch_sinograms = torch.zeros(sheight, len(projection_indices), swidth, device=self.device)
        filled_indices, non_filled_indices, buffer_indices = self.get_filled_indices(projection_indices)
        if len(filled_indices) > 0:
            batch_sinograms[:, filled_indices, :] = (
                self.mem_sinograms[vpu_indices[0], buffer_indices, vpu_indices[2]])
        return batch_sinograms
        # return batch_coordinates, batch_sinograms, batch_indices

    def get_filled_indices(self, projection_indices):
        """
        Vectorized version: returns positions of filled indices in the requested batch.
    
        Args:
            projection_indices: tensor of projection indices (batch)
        Returns:
            filled_indices: positions in the output buffer (batch dim)
            non_filled_indices: positions not filled in the output buffer (batch dim)
            buffer_indices: positions in self.stored_sinograms
        """
        mask = self.all_stored[projection_indices]  # (batch,)
        # indices into output. Max value: len(projection_indices)
        filled_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        non_filled_indices = torch.nonzero(~mask, as_tuple=False).squeeze(1)
        
        # Get the actual global indices that are filled
        # Since filled_indices have a True value in filled_mask, they will find a match in stored_indices
        # So we match asked_filled against self.stored_indices to get buffer indices
        # This assumes that stored_indices are unique and sorted (can be relaxed)
        asked_filled = projection_indices[filled_indices]  # global projection indices
        match = (asked_filled[:, None] == self.mem_indices[None, :]).to(torch.uint8)  # (asked_filled, mem_projections)
        buffer_indices = match.argmax(dim=1)  # index in stored_sinograms
        return filled_indices, non_filled_indices, buffer_indices
    
    def __len__(self):
        return self.num_projections  # the full logical length


class CTDatasetOnDisk(CTDatasetSparse):
    """Lazily loads projections from a folder of per-projection TIFF files (``scan_folder_fmt``
    gives the folder and filename format), using the ``CTDatasetSparse`` buffer as an
    in-memory cache so repeated access to the same projections avoids re-reading from disk.
    On a cache miss, ``__getitem__`` reads the raw TIFF(s) via ``load_projection`` and
    converts them to optical-depth sinogram values via ``scale_projection`` (applying
    flat/dark-field correction if given, otherwise the per-file slope/offset from the TIFF
    description), before caching the result back into the buffer.
    """
    def __init__(self, scan_folder_fmt, projection_times: Union[None, slice, torch.Tensor], device, mem_projections,
                 flat_field = None, dark_field = None, scan_name = None, roi = None, fixed_memory = 0,
                 verbose = False):
        self.scan_folder, self.scan_fmt = scan_folder_fmt
        self.scan_name = scan_name or os.path.basename(self.scan_folder)
        self._init_data_format()
        if projection_times is not None:
            self.projection_times = self.projection_times[projection_times]
        self.projection_times = torch.tensor(self.projection_times, device = device)
        self.use_flat_dark_field = flat_field is not None or dark_field is not None
        self.flat_field = flat_field if flat_field is not None else np.ones((self.height, self.width), dtype = np.float32)
        self.dark_field = dark_field if dark_field is not None else np.zeros((self.height, self.width), dtype = np.float32)
        self.dark_field = torch.as_tensor(self.dark_field, device = device, dtype = torch.float32)
        self.flat_field = torch.as_tensor(self.flat_field, device = device, dtype = torch.float32)
        sino_params = dict(height = self.height, width = self.width,
                           angles = self.projection_times)  # angles only used to know num_projections
        mem_projections = len(self.projection_times) if mem_projections is None else mem_projections
        self.verbose = verbose
        super().__init__(sino_params, self.projection_times, device, mem_projections, roi, fixed_memory)

    def _init_data_format(self):
        """Build the sorted file list for the scan folder, read the first projection's TIFF
        description to derive the slope/offset used to rescale raw pixel values (and hence
        ``self.limits``) and the detector height/width, and set ``projection_times`` to the
        file-derived projection indices (before any ``projection_times`` selection is applied)."""
        self.filelist = ft.filelist(self.scan_folder, self.scan_fmt, sort=True)
        first_image, description = ft.load_tiff(self.filelist[0])
        self.slope, self.offset = float(description.split()[2]), float(description.split()[5])
        self.limits = (self.offset, self.offset + self.slope * np.iinfo(first_image.dtype).max)
        self.height, self.width = first_image.shape
        # projection_times can select files if you don't want to work with the full dataset
        self.projection_times = ft.get_indices(self.filelist)

    def load_projection(self, pi):
        """Read the single raw (uncorrected) projection at file-list index ``pi`` from disk."""
        filename = self.filelist[pi]  # Todo: would need to check if index corresponds in the case of gaps
        projection, description = ft.load_tiff(filename)
        return projection
        
    def __getitem__(self, batch_indices):
        """
        Returns sampled u_values and ground_truth based on batch indices.
        """
        vpu_indices = self._parse_indices(batch_indices)  # Ensure tensor format
        v_indices, projection_indices, u_indices = vpu_indices
        sroi = (v_indices, u_indices)  # select roi
        # selected_indices = self.select_indices(projection_indices)  # possible missing projections?
        selected_indices = self.projection_times[projection_indices]  # TODO check the select_indices functionality isn't missing
        # batch_sinograms = torch.zeros(self.rheight, len(projection_indices), self.rwidth, device = self.device)
        batch_sinograms = super().__getitem__(batch_indices)
        num_preloaded, num_loaded = 0, 0
        for fi, f in enumerate(projection_indices):
            if f in self.mem_indices:  # already filled
                num_preloaded += 1
                continue
            printline = (f'loading {fi - num_preloaded + 1} of {len(selected_indices)} projections'
                         f' from {selected_indices[0]} to {selected_indices[-1]}'
                         f' ({num_preloaded} preloaded)')
            if self.verbose:
                ft.printcounting(printline, fi, len(selected_indices))
            num_loaded += 1
            try:
                pi = selected_indices[fi]
                projection = self.load_projection(pi)
                projection = torch.tensor(projection[self.roi][sroi],
                                          dtype = torch.float32, device = self.device)
                projection = self.scale_projection(projection, sroi)
                batch_sinograms[:,fi,:] = projection
            except (FileNotFoundError, UnboundLocalError):  # indicate that file does not exist (or is corrupted)
                print("Handling missing/corrupted projections by setting them to previous - fix this properly")
                batch_sinograms[:,fi,:] = batch_sinograms[:,fi-1,:]
                projection_indices[fi] = projection_indices[fi-1]
        # store as many as you can in memory
        if num_loaded > 0 and len(projection_indices) > 0:
            super().__setitem__(projection_indices[:self.mem_projections], batch_sinograms[:,:self.mem_projections])
        return batch_sinograms
        # batch_coordinates = self.projection_times[batch_indices]  # Shape: (batch_size,)
        # return batch_coordinates, batch_sinograms, batch_indices

    def scale_projection(self, projection, sroi = (slice(None),)*2):
        """Convert a raw projection into optical-depth sinogram values: if flat/dark-field
        images were supplied, normalize against them; otherwise rescale using the TIFF's
        stored slope/offset. Either way, the result is ``-log`` of the normalized
        transmission (``sroi`` selects the matching sub-region of the flat/dark fields)."""
        if self.use_flat_dark_field:
            normalised = projection - self.dark_field[self.roi][sroi]
            normalised = normalised / (self.flat_field[self.roi][sroi] - self.dark_field[self.roi][sroi])
        else:
            normalised = projection * self.slope + self.offset
        optical_depth = - torch.log(normalised)
        return optical_depth
        
    def select_files_with_indices(self, select):
        """
        NOTE: not currently used/called anywhere in src/ or scripts/.
        Possibly missing projections, this will show as a missing file index.
        Files and indicesare sorted. Select is not
        """
        i = -1
        len_select = len(select)
        len_indices = len(self.projection_times)
        selected = [None] * len_select
        invert_select = ft.listargsort(select)  # calling these on select would sort it
        for si in range(len_select):  # loop over shortest list, this is usually the select list
            s = select[invert_select[si]]  # sorted select
            while i + 1 < len_indices and self.projection_times[i + 1] <= s:  # loop over sorted unique indices
                i += 1  # keep stepping in sorted indices until you catch up with sorted select
            if self.projection_times[i] == select[invert_select[si]]:  # you found a match
                selected[invert_select[si]] = self.filelist[i]
        selected = [f for f in selected if f is not None]
        return selected
    
    def select_files_with_indices2(self, select):
        """ https://discuss.pytorch.org/t/intersection-between-to-vectors-tensors/50364/9 """
        a_cat_b, counts = torch.cat([self.projection_times, select]).unique(return_counts=True)
        intersection = a_cat_b[torch.where(counts.gt(1))].cpu().numpy()
        return intersection


class CTDatasetOnFile(CTDatasetOnDisk):
    """Same lazy-loading/caching/flat-dark-field-correction behaviour as ``CTDatasetOnDisk``,
    but for a single multi-page TIFF file (``scan_file``) instead of one file per projection:
    overrides ``_init_data_format`` to read shape/slope/offset from that file's series, and
    ``load_projection``/``__getitem__`` to index pages of it directly via ``tifffile``
    rather than opening separate files.
    """
    def __init__(self, scan_file, projection_times: Union[None, slice, torch.Tensor], device, mem_projections,
                 scan_name = None, flat_field = None, dark_field = None, roi = None, fixed_memory = 0,
                 verbose = True):
        self.scan_file = scan_file
        scan_folder = os.path.dirname(scan_file)
        self.scan_name = scan_name or os.path.basename(scan_folder)
        super().__init__((scan_folder, None), projection_times, device, mem_projections,
                         flat_field, dark_field, self.scan_name, roi, fixed_memory, verbose)

    def _init_data_format(self):
        """Read shape, dtype and the first page's TIFF description from ``scan_file`` to derive
        the detector height/width and the slope/offset used to rescale raw pixel values."""
        with tifffile.TiffFile(self.scan_file) as tif:
            series = tif.series[0]
            num_projections, self.height, self.width = series.shape
            dtype = series.dtype
            axes = series.axes  # often useful
            description = tif.pages[0].description
        self.slope, self.offset = float(description.split()[2]), float(description.split()[5])
        self.limits = (self.offset, self.offset + self.slope * np.iinfo(dtype).max)
        # projection_times can select files if you don't want to work with the full dataset
        self.projection_times = np.arange(0, num_projections)

    def __getitem__(self, batch_indices):
        """
        Returns sampled u_values and ground_truth based on batch indices.
        """
        vpu_indices = self._parse_indices(batch_indices)  # Ensure tensor format
        v_indices = np.arange(0, self.rheight)[vpu_indices[0]]
        projection_indices = vpu_indices[1].detach().cpu().numpy()
        u_indices = np.arange(0, self.rwidth)[vpu_indices[2]]
        batch_sinograms = self.load_projection(projection_indices)[:,vpu_indices[0], vpu_indices[2]]
        # batch_sinograms = self.load_projection(projection_indices)[:,self.roi][:, v_indices, u_indices]
        # batch_sinograms = self.load_projection((projection_indices, v_indices, u_indices))
        batch_sinograms = torch.as_tensor(batch_sinograms, dtype = torch.float32, device = self.device)
        batch_sinograms = self.scale_projection(batch_sinograms, (v_indices, u_indices))
        if batch_sinograms.ndim == 3:
            batch_sinograms = torch.swapaxes(batch_sinograms, 0, 1)  # vtu order
        return batch_sinograms

    def load_projection(self, pi):
        """Read the projection page(s) at index/indices ``pi`` out of ``scan_file`` without
        loading the whole multi-page TIFF into memory."""
        with tifffile.TiffFile(self.scan_file) as tif:
            series = tif.series[0]
            projection = series.asarray(key=list(pi))  # avoids full load
        return projection


class CTDatasetOnH5File(CTDatasetOnFile):
    """Same idea as ``CTDatasetOnFile`` but backed by a single HDF5 dataset (``scan_file_key``
    is a (file path, dataset key) pair) instead of a TIFF file. ``transpose`` accounts for the
    stored array's last two axes being (width, height) rather than (height, width). Not
    instantiated directly elsewhere in the codebase, but is the base for ``CTDatasetMogno``.
    """
    def __init__(self, scan_file_key, projection_times: Union[None, slice, torch.Tensor], device, mem_projections,
                 scan_name = None, flat_field = None, dark_field = None, roi = None, fixed_memory = 0,
                 verbose = True, transpose = True):
        self.scan_file, self.scan_key = scan_file_key
        scan_folder = os.path.dirname(self.scan_file)
        self.scan_name = scan_name or os.path.basename(self.scan_folder)
        self.transpose = transpose
        CTDatasetOnDisk.__init__(self, (scan_folder, None), projection_times, device, mem_projections,
                                 flat_field, dark_field, self.scan_name, roi, fixed_memory, verbose)

    def _init_data_format(self):
        """Read the HDF5 dataset's shape (accounting for ``transpose``) to derive the detector
        height/width; unlike the TIFF-backed variants there's no slope/offset rescaling
        (``limits`` is just [0, 1])."""
        with h5py.File(self.scan_file, "r") as h5:
            # Transposed?
            if self.transpose:
                num_projections, _, self.width, self.height = h5[self.scan_key].shape
            else:
                num_projections, _, self.height, self.width = h5[self.scan_key].shape
        self.limits = [0,1]
        self.projection_times = np.arange(0, num_projections)

    def load_projection(self, pi):
        """Read projection(s) ``pi`` from the HDF5 dataset, squeezing the singleton second
        axis and swapping the last two axes back if ``transpose`` is set."""
        with h5py.File(self.scan_file, "r") as h5:
            # Transposed?
            projection = h5[self.scan_key][:,0, *pi]  # first squeeze second dimension with length 1
            if self.transpose:
                projection = np.ascontiguousarray(np.swapaxes(projection, -2, -1))
        return projection


class CTDatasetMogno(CTDatasetOnH5File):
    """HDF5-backed dataset for the "Mogno" beamline detector, whose physical sensor is built
    from a 6x6 grid of 256x256-pixel tiles separated by small gaps of dead/missing pixels
    (``gaps_grandes``/``gaps_pequenos``, in pixels, cumulated into
    ``displacement_rows_cumul``/``displacement_colunas_cumul``). The class attributes
    computed at class-definition time (``shifts``, and its inverse ``shifts_inv``) are
    lookup grids mapping between that physical, gapped "tiled" pixel grid (this class's own
    ``height``/``width``) and a contiguous "framed" grid with no gaps, as stored in the raw
    HDF5 data.

    ``tiled_to_pixel_indices``/``tile_projection`` do the actual remapping (with ``np.nan``
    standing in for the gaps); ``__getitem__`` loads only the framed bounding box needed for
    the requested tiled ROI, then tiles and scales it. ``get_mask`` returns a tiled mask of
    which pixels are real detector pixels vs. gaps, e.g. to exclude gaps when comparing
    against reconstructed/simulated data.
    """
    tile_height, tile_width = 256, 256
    tile_rows, tile_cols = 6, 6
    tile_size = (256,256)
    num_tiles = (6,6)
    gaps_grandes = np.array([[0, 49, 49, 50, 49, 49],
                             [0, 49, 49, 50, 48, 49],
                             [0, 49, 49, 50, 49, 49],
                             [0, 49, 49, 50, 49, 49],
                             [0, 49, 49, 50, 49, 49],
                             [0, 49, 49, 51, 49, 49]])  # widest row
    #                         ^ displacement_rows

    gaps_pequenos = np.array([[1, 1, 1, 0, 0, 0],  # displacement_colunas
                              [3, 3, 3, 4, 3, 3],
                              [3, 3, 3, 3, 3, 3],
                              [3, 4, 4, 3, 4, 3],
                              [3, 3, 3, 3, 3, 3],
                              [3, 3, 3, 3, 3, 3]])
    #                             ^ highest column

    displacement_rows_cumul = np.cumsum(gaps_grandes, axis = 1)
    displacement_colunas_cumul = np.cumsum(gaps_pequenos, axis = 0)

    height = tile_rows * tile_height + np.max(displacement_colunas_cumul[-1,:])
    width = tile_cols * tile_width + np.max(displacement_rows_cumul[:,-1])

    # shift from pixel grid to tiled detector grid
    shifts = np.meshgrid(np.arange(0, tile_rows * tile_height),
                         np.arange(0, tile_cols * tile_width), indexing='ij')
    for ti, tj in np.ndindex(tile_rows, tile_cols):
        shifts[0][ti * tile_height: (ti + 1) * tile_height,
                  tj * tile_width: (tj + 1) * tile_width] += displacement_colunas_cumul[ti, tj]
        shifts[1][ti * tile_height: (ti + 1) * tile_height,
                  tj * tile_width: (tj + 1) * tile_width] += displacement_rows_cumul[ti, tj]

    # shift from tiled detector grid to pixel grid
    shifts_inv = np.zeros((2, height, width), dtype=np.float32) + np.nan
    for ti, tj in np.ndindex(tile_rows, tile_cols):
        drc = displacement_rows_cumul[ti, tj]
        dcc = displacement_colunas_cumul[ti, tj]
        shifts_inv[0, dcc + ti * tile_height: dcc + (ti + 1) * tile_height,
                   drc + tj * tile_width: drc + (tj + 1) * tile_width] = np.arange(ti * tile_height, (ti + 1) * tile_height)[:,np.newaxis]
        shifts_inv[1, dcc + ti * tile_height: dcc + (ti + 1) * tile_height,
                   drc + tj * tile_width: drc + (tj + 1) * tile_width] = np.arange(tj * tile_width, (tj + 1) * tile_width)[np.newaxis,:]

    def __init__(self, scan_file_key, projection_times: Union[None, slice, torch.Tensor], device, mem_projections,
                 scan_name = None, flat_field = None, dark_field = None, roi = None, fixed_memory = 0,
                 verbose = True, transpose = True):
        """Load the scan, then tile any given flat/dark-field images from the framed grid into
        the gapped tiled grid (temporarily setting ``roi`` to ``None`` so the full field gets
        tiled), before restoring the requested ``roi``."""
        super().__init__(scan_file_key, projection_times, device, mem_projections,
                         scan_name, flat_field, dark_field, roi, fixed_memory, verbose, transpose)
        # Set roi to max extent so the full flat field is tiled to a frame
        self.roi = None
        if flat_field is not None:
            self.flat_field = torch.as_tensor(self.tile_projection(flat_field, gap = np.nan),
                                              device=device, dtype=torch.float32)
        if dark_field is not None:
            self.dark_field = torch.as_tensor(self.tile_projection(dark_field, gap = np.nan),
                                              device=device, dtype=torch.float32)
        self.roi = roi

    def _init_data_format(self):
        """Read only the number of projections from the HDF5 file; height/width and pixel
        scaling for this detector come from the class-level tile geometry instead."""
        with h5py.File(self.scan_file, "r") as h5:
            num_projections = len(h5[self.scan_key])
        self.limits = [0,1]
        self.projection_times = np.arange(0, num_projections)

    def tiled_to_pixel_indices(self, v_indices, u_indices):
        """Map (v, u) coordinates in the gapped tiled grid to their corresponding coordinates
        in the contiguous framed pixel grid, via the precomputed ``shifts_inv`` lookup.
        Coordinates that fall in a gap (no real pixel) come back as ``np.nan``."""
        # start with v_indices in [0, 1553] and u_indices in [0, 1783]
        # get pixel indices in [0, 1536] or np.nan
        tiled_indices = np.meshgrid(v_indices, u_indices, indexing = 'ij')
        pixel_indices = self.shifts_inv[:, *tiled_indices]
        return pixel_indices

    def tile_projection(self, projection, gap = np.nan, sroi =(slice(None),) * 2, load_roi = (slice(None),) * 2):
        """Remap a projection (or batch of projections) from the contiguous framed pixel grid
        into the gapped tiled detector grid, filling gaps with ``gap``.

        Args:
            projection: framed-grid projection(s), cropped to ``load_roi`` (as loaded from the
                HDF5 file - only the bounding box actually needed has to be loaded).
            sroi: the (v, u) sub-region of the tiled grid to produce (relative to ``self.roi``).
            load_roi: the (v, u) region of the framed grid that ``projection`` covers, used to
                offset into it correctly.
        """
        v0 = load_roi[0].start if load_roi[0].start is not None else 0
        u0 = load_roi[1].start if load_roi[1].start is not None else 0
        # create ROI projection
        tiled_projection = gap + np.zeros((self.height, self.width), dtype=projection.dtype)[self.roi][sroi]
        v_coords = np.arange(0, self.height)[self.roi[0]][sroi[0]]
        u_coords = np.arange(0, self.width)[self.roi[1]][sroi[1]]
        if projection.ndim == 3:
            tiled_projection = np.tile(tiled_projection, (projection.shape[0], 1, 1))
        # loop over tiles
        for ti, tj in np.ndindex(self.tile_rows, self.tile_cols):
            dv = self.displacement_colunas_cumul[ti, tj]  # - self.roi[0].start
            du = self.displacement_rows_cumul[ti, tj]  # - self.roi[1].start
            # find pixels in overlap between framed tile and sroi
            tiled_v_coords = v_coords[((dv + ti * self.tile_height <= v_coords) &  # min v
                                       (v_coords < dv + (ti + 1) * self.tile_height))]  # max v
            tiled_u_coords = u_coords[((du + tj * self.tile_width <= u_coords) &  # min u
                                       (u_coords < du + (tj + 1) * self.tile_width))]  # max u
            if len(tiled_u_coords) == 0 or len(tiled_v_coords) == 0:
                continue
            # pixels bounded by framed tiles always have a counterpart, so not nan
            tiled_selection = np.meshgrid(tiled_v_coords - v_coords.item(0), tiled_u_coords - u_coords.item(0),
                                          indexing = 'ij')
            pixel_coords = self.tiled_to_pixel_indices(tiled_v_coords, tiled_u_coords)
            pixel_coords = np.int32(pixel_coords)
            pixel_selection = projection[..., pixel_coords[0] - v0, pixel_coords[1] - u0]
            tiled_projection[..., *tiled_selection] = pixel_selection

        return tiled_projection

    def __getitem__(self, batch_indices):
        """
        Returns sampled u_values and ground_truth based on batch indices.
        """
        vpu_indices = self._parse_indices(batch_indices)  # Ensure tensor format
        v_indices = np.arange(0, self.height)[self.roi[0]][vpu_indices[0]]
        projection_indices = vpu_indices[1]
        u_indices = np.arange(0, self.width)[self.roi[1]][vpu_indices[2]]
        pixel_coordinates = self.tiled_to_pixel_indices(v_indices, u_indices)

        # Load the smallest bounding box that tiles to the selected roi
        load_roi = (slice(int(np.nanmin(pixel_coordinates[0])), 1+int(np.nanmax(pixel_coordinates[0]))),
                    slice(int(np.nanmin(pixel_coordinates[1])), 1+int(np.nanmax(pixel_coordinates[1]))))
        sroi = (vpu_indices[0], vpu_indices[2])
        with h5py.File(self.scan_file, "r") as h5:
            # Transposed?
            if self.transpose:
                batch_sinograms = h5[self.scan_key][projection_indices, 0, load_roi[1], load_roi[0]]
                batch_sinograms = np.ascontiguousarray(np.swapaxes(batch_sinograms, -2, -1))
            else:
                batch_sinograms = h5[self.scan_key][projection_indices, 0, load_roi[0], load_roi[1]]
        batch_sinograms = self.tile_projection(batch_sinograms, load_roi = load_roi, sroi = sroi)
        batch_sinograms = torch.as_tensor(batch_sinograms, dtype = torch.float32, device = self.device)
        batch_sinograms = self.scale_projection(batch_sinograms, sroi = sroi)
        if batch_sinograms.ndim == 3:
            batch_sinograms = torch.swapaxes(batch_sinograms, 0, 1)  # vtu order
        batch_sinograms = batch_sinograms.contiguous()
        return batch_sinograms

    def get_mask(self, roi = None):
        """Return a tiled-grid mask (1 where a real detector pixel exists, gap/NaN elsewhere)
        for the given ``roi``, e.g. to exclude the physical detector's tile gaps when
        comparing against reconstructed/simulated data."""
        # Set mask to full extent
        self_roi = self.roi
        self.roi = roi
        mask = np.ones((self.tile_rows * self.tile_height,
                       self.tile_cols * self.tile_width), dtype=np.float32)
        tiled_mask = self.tile_projection(mask)
        self.roi = self_roi
        return tiled_mask


class CTDatasetDifference(CTDataset):
    """Decorator-style dataset that wraps two other ``CTDataset``s and returns/writes their
    difference rather than sourcing data of its own: ``ref_dataset`` is treated as a single
    (typically static) rotation and its projection indices are wrapped (modulo its own
    ``num_projections``) so it can be reused across the full, typically longer,
    ``dyn_dataset``. ``__getitem__`` returns ``dyn_dataset - ref_dataset``; ``__setitem__``
    updates ``dyn_dataset`` so that the difference matches the given values.
    """
    def __init__(self, ref_dataset: CTDataset, dyn_dataset: CTDataset, scan_name = None):
        self.ref_dataset = ref_dataset
        self.dyn_dataset = dyn_dataset  # type: CTDataset
        self.scan_name = scan_name or (self.dyn_dataset.scan_name + '_diff')
        sino_params = dyn_dataset.sino_params  # type: dict
        sino_params['num_projections'] = dyn_dataset.num_projections
        sino_params['roi'] = dyn_dataset.roi
        self.ref_dataset.roi = dyn_dataset.roi
        CTDataset.__init__(self, sino_params = sino_params, projection_times = dyn_dataset.projection_times)

    @property
    def scan_folder(self):
        """Proxy for ``dyn_dataset.scan_folder`` (only valid when ``dyn_dataset`` is disk-backed)."""
        assert isinstance(self.dyn_dataset, CTDatasetOnDisk)
        return self.dyn_dataset.scan_folder

    @scan_folder.setter
    def scan_folder(self, folder):
        """Cannot actually redirect the underlying (proxied) scan folder; just warns instead."""
        print(f"Attempting to adjust scan folder from {self.scan_folder} to {folder}, check this behaviour")

    def __getitem__(self, vpu_indices):
        """Return ``dyn_dataset``'s sample minus the corresponding ``ref_dataset`` sample,
        wrapping the requested projection indices modulo ``ref_dataset.num_projections``
        since the reference is a single (shorter) rotation reused across ``dyn_dataset``."""
        v_indices, projection_indices, u_indices = self._parse_indices(vpu_indices)
        # reference dataset is a single rotation, so wrap the projection indices
        ref_projection_indices = projection_indices % self.ref_dataset.num_projections
        ref_sinogram_sample = self.ref_dataset[v_indices, ref_projection_indices, u_indices]
        dyn_sinogram_sample = self.dyn_dataset[vpu_indices]
        diff_sinogram_sample = dyn_sinogram_sample - ref_sinogram_sample.to(dyn_sinogram_sample.device)
        return diff_sinogram_sample

    def __setitem__(self, vpu_indices, sinogram_slices):
        """
        Update dynamic dataset

        :param vpu_indices:
        :param sinogram_slices:
        :return:
        """
        v_indices, projection_indices, u_indices = self._parse_indices(vpu_indices)
        # reference dataset is a single rotation, so wrap the projection indices
        ref_projection_indices = projection_indices % self.ref_dataset.num_projections
        ref_sinogram_sample = self.ref_dataset[v_indices, ref_projection_indices, u_indices]
        self.dyn_dataset[vpu_indices] = ref_sinogram_sample.to(sinogram_slices.device) + sinogram_slices


class CTDatasetAverage(CTDataset):
    """Decorator-style dataset that wraps an underlying ``CTDataset`` and averages every
    ``averages`` (must be odd) consecutive raw projections together into a single output
    projection - temporal binning along the projection axis that trades angular sampling
    for lower noise and a smaller effective ``num_projections``. ``__getitem__`` expands each
    requested (averaged) projection index back into its underlying window via
    ``_get_expanded_indices`` and averages over that window; ``__setitem__`` does the reverse
    by duplicating written values across the whole window (no true de-averaging/
    interpolation).
    """
    def __init__(self, dataset: CTDataset, averages = 3):
        self.dataset = dataset
        self.averages = averages
        assert averages % 2 == 1, "Haven't considered even averaging yet, might already work"

        projection_times = dataset.projection_times
        self.projection_times = projection_times[averages//2::averages]

        self.half_window = (averages - 1) // 2
        self.offsets = torch.arange(-self.half_window, self.averages - self.half_window, dtype=torch.long, device=dataset.device)

        sino_params = dataset.sino_params  # type: dict
        sino_params['num_projections'] = len(self.projection_times)
        sino_params['roi'] = dataset.roi
        CTDataset.__init__(self, sino_params = sino_params, projection_times = self.projection_times)

    def update_sino_params_average(self, sino_params):
        """Adjust a ``sino_params`` dict describing the underlying (un-averaged) dataset so it
        instead describes this averaged dataset.

        Args:
            sino_params: params for the underlying dataset; must contain an ``'angles'``
                array aligned with ``self.dataset.projection_times``.
        Returns:
            sino_params, updated in place with ``'angles'``/``'last_angle'``/
            ``'num_projections'`` restricted to the averaged projection set.
        """
        mask = torch.isin(self.dataset.projection_times, self.projection_times)
        indices = torch.where(mask)[0]
        updated_angles = sino_params['angles'][indices.cpu().numpy()]
        sino_params['angles'] = updated_angles
        sino_params['last_angle'] = updated_angles[-1]
        sino_params['num_projections'] = len(self.projection_times)
        return sino_params

    def convert_disabled_projections(self, disabled_projections):
        """Map indices of disabled projections expressed in the underlying (un-averaged)
        dataset's index space to the corresponding indices in this averaged dataset."""
        disabled_projection_values = self.dataset.projection_times[disabled_projections]
        mask_to_disable = torch.isin(self.projection_times,disabled_projection_values)
        disabled_indices_averaged = torch.where(mask_to_disable)[0]
        return disabled_indices_averaged.cpu().tolist()

    def _get_expanded_indices(self, projection_indices):
        """Expand each requested averaged-output projection index into the full window of
        ``self.averages`` underlying raw projection indices it is computed from."""
        projection_indices = projection_indices * self.averages + self.half_window
        expanded_blocks = projection_indices.unsqueeze(1) + self.offsets.unsqueeze(0)
        expanded_indices = expanded_blocks.flatten()
        return expanded_indices

    def __getitem__(self, vpu_indices):
        """Load the underlying dataset's raw projections for the expanded index window and
        average each group of ``self.averages`` of them into one output projection."""
        v_indices, projection_indices, u_indices = self._parse_indices(vpu_indices)

        # add surrounding indices to projection_indices
        expanded_indices = self._get_expanded_indices(projection_indices)

        # load indices from sinogram
        sinogram_sample = self.dataset[v_indices, expanded_indices, u_indices]

        # average
        reshaped_sample = sinogram_sample.reshape(sinogram_sample.shape[0], len(projection_indices),self.averages,sinogram_sample.shape[2])
        averaged_sample = torch.mean(reshaped_sample, dim=2)

        return averaged_sample

    def __setitem__(self, vpu_indices, sinogram_slices):
        """
        Update dataset - just duplicate underlying over averaged projections #TODO interpolate?

        :param vpu_indices:
        :param sinogram_slices:
        :return:
        """
        v_indices, projection_indices, u_indices = self._parse_indices(vpu_indices)
        expanded_indices = self._get_expanded_indices(projection_indices)
        duplicated_slices = torch.repeat_interleave(sinogram_slices, repeats=self.averages, dim=1)
        self.dataset[v_indices, expanded_indices, u_indices] = duplicated_slices
