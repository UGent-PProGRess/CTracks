"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

This file defines the entities that are used in the vtools package.

The entities are:
- ParticleIterator

--------------------------------------------------------------------------
Role in the data flow
--------------------------------------------------------------------------
A `ParticleIterator` is the particle-iteration step (see `plot_3d_vectors.py` for
the full pipeline: mesh generation -> particle iteration -> scene assembly ->
render/save). It wraps a source of per-frame (position, velocity) arrays - a
linked-trajectories CSV/DataFrame for `ParticleIterator_DF`, or raw `scan_{i}.npy`
reconstruction files (via `utils.convert_np_to_df`) for `ParticleIterator_CTrack` -
and turns each frame, on demand (`iterator[frame_idx]`), into a pyvista glyph mesh:
velocity-oriented, magnitude-colored arrows, one per particle, sized between
`arrow_lim` and restricted to the `velocity_split` magnitude range. `Explorer3D`
(see `Explorer3D.py`) is the caller that indexes into these iterators per frame to
assemble/update the actual 3D scene.
"""
import pandas as pd
import numpy as np
import pyvista as pv
from abc import ABC, abstractmethod

from .utils import convert_np_to_df


class ParticleIterator(ABC):
    """Abstract base for per-frame particle-velocity-glyph providers.

    Subclasses supply raw (positions, velocities) per frame via
    `get_particle`; this base class turns that into a pyvista glyph mesh
    (arrows oriented/colored by velocity) via `get_glyph`, and exposes it
    through indexing (`iterator[frame_idx]`).
    """
    def __init__(self,
                 name,
                 arrow_lim=(0.1, 3),
                 show_arrowhead = True,
                 velocity_split = (None, None)):
        self.name = name
        self.arrow_min, self.arrow_max = arrow_lim
        self.show_arrowhead = show_arrowhead
        self.velocity_split = velocity_split

    def compute_velocity_magnitudes(self, velocities):
        """
        Compute the vector magnitudes (L2 norm) of velocity vectors.
        velocities: shape (N, 3)
        returns: shape (N,) with each magnitude
        """
        return np.linalg.norm(velocities, axis=1)

    def map_magnitudes_to_size(self, magnitudes, low, high):
        """
        Map magnitudes [min, max] -> [low, high] (linear mapping/clamping)
        If you just want to clamp, you can use np.clip directly.
        """
        # First find global min & max from your data or from magnitudes
        mag_min = magnitudes.min()
        mag_max = magnitudes.max()
        # Avoid division by zero
        denom = max(mag_max - mag_min, 1e-12)

        # Linear mapping
        scaled = low + (magnitudes - mag_min) * (high - low) / denom

        # Clamp to [low, high] just in case
        scaled = np.clip(scaled, low, high)
        return scaled

    @abstractmethod
    def get_particle(self, index):
        """Return (positions, velocities) numpy arrays, each shape (N, 3), for frame `index`."""
        pass

    def get_glyph(self, index):
        """
        Build a pyvista glyph mesh (velocity-oriented/colored arrows, one
        per particle) for frame `index`, using `get_particle` for the raw
        data. Particles whose velocity magnitude falls outside
        `self.velocity_split` are excluded.
        """
        arrow_min = self.arrow_min
        arrow_max = self.arrow_max

        positions, velocities = self.get_particle(index)
        magnitudes = self.compute_velocity_magnitudes(velocities)
        arrow_sizes = self.map_magnitudes_to_size(
            magnitudes, arrow_min, arrow_max)

        low = self.velocity_split[0] if self.velocity_split[0] is not None else -np.inf
        high = self.velocity_split[1] if self.velocity_split[1] is not None else np.inf

        mask = (magnitudes >= low) & (magnitudes <= high)

        points = pv.PolyData(positions[mask])
        points['velocity'] = velocities[mask]
        points["mags"] = magnitudes[mask]            # For coloring
        points["arrowScale"] = arrow_sizes[mask]     # For sizing the glyphs

        points.set_active_scalars("mags")
        if self.show_arrowhead:
            arrow = pv.Arrow()
        else:
            arrow = pv.Arrow(tip_length=0, tip_radius = 0, shaft_radius = 0.05)
        glyphs = points.glyph(
            orient='velocity',
            scale='arrowScale',
            color_mode='scalar',
            factor=15,
            geom=arrow)
        glyphs.set_active_scalars("mags")
        return glyphs

    @abstractmethod
    def __len__(self):
        """
        Returns the total number of frames available.
        """
        pass

    def __getitem__(self, index):
        """
        Supports indexing and slicing.
        """
        return self.get_glyph(index)

    # NOTE: this class does not implement __iter__/__next__. A previous
    # version did, but referenced self.fluid_files/self.get_velocity_field,
    # which are never defined anywhere in this hierarchy (confirmed dead via
    # grep - nothing in the codebase iterates a ParticleIterator directly;
    # all call sites use __getitem__ indexing instead). Removed rather than
    # left in place to avoid presenting a broken public API.


class ParticleIterator_DF(ParticleIterator):
    """Loads (position, velocity) tracks per frame from a linked-trajectories
    DataFrame/CSV (columns: frame, x, y, z, vx, vy, vz by default).
    """
    def __init__(self,
                 name,
                 df_path,
                 frame_key='frame',
                 x_key='x',
                 y_key='y',
                 z_key='z',
                 vx_key='vx',
                 vy_key='vy',
                 vz_key='vz',
                 shift_array=np.array([0, 0, 0]).reshape(-1, 3),
                 frame_start=0,
                 frame_end=10000,
                 **kwargs
                 ):
        """Load `df_path` and keep only rows with `frame_start <= frame_key <= frame_end`.

        `shift_array` is added to every position (see `get_particle`), e.g.
        to compensate for a crop applied to the corresponding pore-structure
        volume. `x_key`/.../`vz_key` name the position/velocity columns.
        """
        super().__init__(name, **kwargs)
        self.df = pd.read_csv(df_path)
        self.df = self.df[(self.df[frame_key] >= frame_start) &
                          (self.df[frame_key] <= frame_end)]
        self.frame_key = frame_key
        self.shift = shift_array
        self.x_key = x_key
        self.y_key = y_key
        self.z_key = z_key
        self.vx_key = vx_key
        self.vy_key = vy_key
        self.vz_key = vz_key

    def __len__(self):
        """Number of distinct frames available."""
        return self.df[self.frame_key].nunique()

    def get_frame(self, index):
        """Return the sub-DataFrame of rows belonging to frame `index`."""
        df_frame = self.df[self.df[self.frame_key] == index].copy()
        return df_frame

    def get_particle(self, index):
        """Return (positions, velocities) numpy arrays for frame `index`, shifted by `self.shift`."""
        df_frame = self.get_frame(index)
        positions = df_frame[
            [self.x_key, self.y_key, self.z_key]].to_numpy() + self.shift
        velocities = df_frame[
            [self.vx_key, self.vy_key, self.vz_key]].to_numpy()

        return positions, velocities


class ParticleIterator_CTrack(ParticleIterator_DF):
    """`ParticleIterator_DF` sourced from raw `scan_{i}.npy` reconstruction
    result files instead of a pre-linked CSV - converts them via
    `convert_np_to_df` first, then delegates to `ParticleIterator_DF`.
    """

    def __init__(self,
                 name,
                 np_files,
                 track_set = 'reconstruction',
                 shift_array=np.array([0, 0, 0]).reshape(-1, 3),
                 frame_start=0,
                 frame_end=10000,
                 **kwargs):
        """`np_files` is a list of `scan_{i}.npy` file paths (one per frame);
        `track_set` selects which entry of each file's results dict to use
        (e.g. "reconstruction" or "ground truth").
        """
        df_path = convert_np_to_df(np_files, track_set)
        super().__init__(name, df_path, shift_array=shift_array, frame_start=frame_start, frame_end=frame_end, **kwargs)




