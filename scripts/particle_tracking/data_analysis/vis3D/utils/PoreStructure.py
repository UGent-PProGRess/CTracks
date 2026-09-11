"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

This file defines the entities that are used in the vtools package.

The entities are:
- PoreStructure
- FluidIterator
- ParticleIterator

--------------------------------------------------------------------------
Role in the data flow
--------------------------------------------------------------------------
`PoreStructure_CT` is the mesh-generation step (see `plot_3d_vectors.py` for the
full pipeline: mesh generation -> particle iteration -> scene assembly -> render/save).
It loads a segmented mask TIFF and, on `get_surface()`, extracts and smooths a static
pyvista surface mesh from it (via `tif_2_geo`/`geo_2_mesh` in `utils.py`, marching
cubes under the hood) - this mesh does not change per frame, unlike the particle
velocity glyphs built by `Particle.py`.
"""
from abc import ABC, abstractmethod

import numpy as np
from skimage import io
from .utils import tif_2_geo, geo_2_mesh


class PoreStructure(ABC):
    """
    Abstract base class that represents the pore structure of a porous media.
    """
    @abstractmethod
    def get_geo(self):
        """
        Abstract method that should return the geometry of the pore structure.
        This includes vertices and faces.
        """
        pass

    @abstractmethod
    def get_surface(self):
        """
        Abstract method that should return a mesh representing the pore
        surface.
        """
        pass


class PoreStructure_CT(PoreStructure):
    """
    Concrete implementation of PoreStructure for CT scan data.
    First load the tif data, then convert it to a mesh.
    Apply smoothing to the mesh if necessary.
    """
    def __init__(self,
                 tif_file,
                 threshold=0,
                 down_sample_factor=4,
                 smooth_iter=10,
                 smooth_factor=0.5,
                 scale=None,
                 expand_distance=10,
                 permute_axes=None,
                 slicer=None):
        """Load `tif_file` as a labeled/segmented 3D volume.

        Parameters
        ----------
        tif_file : str
            Path to the segmented mask TIFF (voxel value `threshold` = pore space).
        threshold : scalar
            Voxel value identifying the surface to extract (see `tif_2_geo`).
        down_sample_factor : int
            Downsampling factor applied before marching cubes.
        smooth_iter, smooth_factor : passed to `geo_2_mesh` (see its docstring
            for a caveat: currently not actually applied there).
        scale : float or None
            If given, uniformly scales the extracted mesh vertices.
        expand_distance : float
            Unused by `get_geo`/`get_surface` below; kept for API compatibility.
        permute_axes : tuple or None
            If given, reorders the (x, y, z) vertex columns, e.g. to align
            with a downstream plotting convention.
        slicer : tuple of slice or None
            If given, applied to `tif_data` before geometry extraction.
        """
        self.tif_data = io.imread(tif_file)
        self.threshold = threshold
        self.down_sample_factor = down_sample_factor
        self.smooth_iter = smooth_iter
        self.smooth_factor = smooth_factor
        self.scale = scale
        self.permute_axes = permute_axes
        self.slicer = slicer
        self.expand_distance = expand_distance

    def get_geo(self):
        """Extract (and optionally slice/scale/permute) the surface geometry
        of the pore structure. Returns (verts, faces).
        """
        if self.slicer:
            self.tif_data = self.tif_data[self.slicer]
        verts, faces = tif_2_geo(
            self.tif_data,
            threshold=self.threshold,
            down_sample_factor=self.down_sample_factor
        )
        if self.scale:
            verts *= self.scale
        if self.permute_axes:
            verts = verts[:, self.permute_axes]
        return verts, faces

    def get_surface(self):
        """Return a smoothed pyvista mesh of the pore surface (see `get_geo`, `geo_2_mesh`)."""
        verts, faces = self.get_geo()
        mesh_surface = geo_2_mesh(
            verts, faces,
            smooth_iter=self.smooth_iter)
        mesh = mesh_surface

        return mesh

    def crop_update_shift(self, crop_slice, current_shift=np.array((0, 0, 0))):
        """Crop `self.tif_data` in place by `crop_slice` and return the
        updated particle-position shift that compensates for it.

        crop_slice : tuple of slice, in (z, y, x) order (matching tif_data).
        current_shift : existing shift to compose with, in (x, y, z) order.
        """
        self.tif_data = self.tif_data[crop_slice]

        # crop_slice is (z, y, x) but the shift is (x, y, z), hence the
        # [::-1]; shift is *added* to positions elsewhere, so the crop
        # offset must be subtracted here (hence the -1).
        crop_offsets = np.array([-1 * s.start if s.start is not None else 0 for s in
                                 crop_slice[::-1]])
        total_shift = current_shift + crop_offsets
        total_shift = total_shift.reshape(-1, 3)
        return total_shift
