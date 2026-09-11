"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

A Visulisation script utilising pyvista API to compare the physics inside
a porous media.

A 3D scene with two subplots is created to compare the particle tracking
between the ground truth and the prediction.

Two view are linked to facilitate the comparison.

--------------------------------------------------------------------------
Data flow
--------------------------------------------------------------------------
1. Mesh generation: `PoreStructure_CT(seg_path, ...)` (see `utils/PoreStructure.py`)
   loads the segmented pore/mask TIFF and turns it into a static pyvista surface mesh
   (marching cubes + smoothing) on demand.
2. Particle iteration: `load_particles_df` (linked-trajectories CSV) or
   `load_particles_ctracks`/`load_particles_ctracks_dual` (raw `scan_*.npy`-style
   reconstruction files) wrap the source in one or two `ParticleIterator_*` instances
   (see `utils/Particle.py`) - "one or two" because a `velocity_split` threshold, if
   given, splits particles into separate "low"/"high" velocity-magnitude iterators so
   they can be styled differently. Each iterator turns a frame index into a pyvista
   glyph mesh (velocity-oriented/colored arrows) on demand via `__getitem__`.
3. Scene assembly: `setup_plotter`/`setup_plotter_duo` build a `pyvista.Plotter`, wrap
   the pore mesh and particle iterator(s) in an `Explorer3D` (see `utils/Explorer3D.py`)
   - which owns the pyvista actors and knows how to (re)build/update them per frame -
   and call `explorer.set_scene3d(0)` to add everything to the scene, then position the
   camera via `setup_cam`.
4. Render/save - four entry points, two single-panel and two dual-panel:
   - `plot_single`/`plot_video`: headless/production use (analyse_porous.py,
     analyse_simulation.py) - show the scene once (or screenshot it off-screen), or
     loop over frames via `Explorer3D.auto_animation` (swaps actor data in place) and
     screenshot each frame to `save_folder`.
   - `plot_interactive`/`plot_duo`: interactive exploration use
     (`data_exploration/single_view.py`, `data_exploration/duo_view.py`) - open an
     interactive window with a frame slider (`plot_explorer`), or save the scrubbed
     animation as one GIF (`save_explorer`); `plot_duo` additionally links two panels.
"""

import glob
import os
import re

import numpy as np
import pyvista as pv
from scripts.particle_tracking.data_analysis.vis3D.utils.Explorer3D import Explorer3D
from scripts.particle_tracking.data_analysis.vis3D.utils.PoreStructure import PoreStructure_CT
from scripts.particle_tracking.data_analysis.vis3D.utils.Particle import ParticleIterator_DF, ParticleIterator_CTrack

pv.global_theme.allow_empty_mesh = True

def load_particles_df(file_path, total_shift=np.array((0, 0, 0)).reshape(-1, 3), arrow_lim=(1, 5),
                   plot_all_tracks = True, velocity_split=None, show_arrowhead = False, frame_end = 10000, plot_low=True):
    """Build one or two `ParticleIterator_DF` instances from a linked-trajectories CSV.

    If `velocity_split` is given (and `plot_low` is True), the particles are
    split into a "low" and "high" velocity-magnitude iterator (split at
    `velocity_split`, each with its own arrow-size range) so they can be
    styled differently; otherwise a single iterator covering all particles
    is returned. If `plot_all_tracks` is set, every particle's frame is
    overwritten to 0 so all tracks are drawn simultaneously as a static
    trace rather than animated per-frame.

    Returns (iterators, num_frames).
    """
    particle_high_iterator = ParticleIterator_DF(
        "particle_high_tp",
        file_path,
        shift_array=total_shift,
        arrow_lim=arrow_lim,
        velocity_split=(velocity_split, None),
        frame_end=frame_end-1,
    )
    num_frames = particle_high_iterator.df[particle_high_iterator.frame_key].nunique()
    if plot_all_tracks:
        particle_high_iterator.df[particle_high_iterator.frame_key] = 0  # set all vectors to be the same frame
    particle_high_iterator.show_arrowhead = show_arrowhead

    if velocity_split is None or plot_low is False: return [particle_high_iterator], num_frames

    particle_low_iterator = ParticleIterator_DF(
        "particle_low_tp",
        file_path,
        shift_array=total_shift,
        arrow_lim=(arrow_lim[0] / 3, arrow_lim[1] / 3),
        velocity_split=(None, velocity_split),
        frame_end=frame_end-1,
    )
    if plot_all_tracks:
        particle_low_iterator.df[particle_low_iterator.frame_key] = 0  # set all vectors to be the same frame
    particle_low_iterator.show_arrowhead = show_arrowhead

    return [particle_low_iterator, particle_high_iterator], num_frames


def setup_plotter(particle_iterator_ct, rock_surface, show_clip_panel,
                  crop_bounds = None, clim = (1,7), show_labels = False, plot_grains = False,
                  surface_transparency = 0.05, interactive = True, cam_params = None, cbar_params = None):
    """Create a single-panel pyvista Plotter with the pore surface + velocity glyphs, and set up the camera.

    Returns (plotter, Explorer3D instance).
    """
    plotter = pv.Plotter(
        title="Particle streamlines",
        shape=(1, 1),
        window_size=[1400, 1000],
        off_screen=not interactive)

    explorer_ct = Explorer3D(
        velocity_iterators=particle_iterator_ct,
        pore_structure=rock_surface,
        num_frames=len(particle_iterator_ct[0]),
        plotter=plotter,
        clip_panel=show_clip_panel,
        clim = clim,
        crop_bounds = crop_bounds,
        plot_grains = plot_grains,
        surface_transparency = surface_transparency,
        cbar_params = cbar_params
    )

    plotter.show_grid(
        all_edges=False,
        show_xlabels=show_labels,
        show_ylabels=show_labels,
        show_zlabels=show_labels,
    )

    explorer_ct.set_scene3d(0)

    setup_cam(cam_params, plotter)

    return plotter, explorer_ct


def setup_cam(cam_params, plotter):
    """Reset and position the camera on `plotter`, applying `cam_params` (azimuth/elevation/focal_shift/zoom) or a default view."""
    if cam_params is None:
        cam_params = {
            "azimuth": 90,
            "elevation": 15,
            "focal_shift": [-50, 0, -30],
            "zoom": 1.4
        }
    plotter.remove_bounds_axes()
    plotter.reset_camera()
    plotter.camera_position = "xz"
    plotter.camera.azimuth = cam_params['azimuth']
    plotter.camera.elevation = cam_params['elevation']
    current_focal = list(plotter.camera.focal_point)
    current_focal[0] += cam_params['focal_shift'][0]  # Shift focal point 'left' (moves mesh right)
    current_focal[1] += cam_params['focal_shift'][1]
    current_focal[2] += cam_params['focal_shift'][2]  # Shift focal point 'down' (moves mesh up)
    plotter.camera.focal_point = current_focal
    plotter.camera.zoom(cam_params['zoom'])



def plot_single(seg_path, ct_file,  scale = 1, down_sample_factor = 4, crop_bounds = None,
            shift=np.array((0,0,0)).reshape(-1, 3), cam_params = None, cbar_params = None, arrow_lim = (1,5),
            clim = (1,7), show_clip_panel = False, plot_grains = False, surface_transparency = 0.05,
            velocity_split=None, show_arrowhead=False, plot_low = True, save_file = None, interactive = True):
    """Render a single static 3D scene: pore surface (from `seg_path`) plus
    all particle-track velocity glyphs (from `ct_file`, a linked-trajectories
    CSV) overlaid together (via `load_particles_df(..., plot_all_tracks=True)`
    inside `setup_plotter`).

    If `interactive`, opens an interactive pyvista window; otherwise renders
    off-screen and saves a screenshot to `save_file` (if given).
    """
    rock_surface = PoreStructure_CT(
        seg_path,  # noqa
        scale=scale,
        down_sample_factor=down_sample_factor,
        permute_axes=(2, 1, 0))

    particle_iterator_ct, num_ct_frames = load_particles_df(ct_file, shift,
                                                  arrow_lim=arrow_lim,
                                                  velocity_split=velocity_split,
                                                  show_arrowhead=show_arrowhead,
                                                  plot_low=plot_low)

    plotter, explorer = setup_plotter(particle_iterator_ct, rock_surface, show_clip_panel, crop_bounds = crop_bounds,
                                clim = clim, show_labels=False, plot_grains = plot_grains,
                                surface_transparency = surface_transparency, interactive = interactive,
                                cam_params = cam_params, cbar_params = cbar_params)
    if interactive:
        plotter.show()
    elif save_file is not None:
        plotter.screenshot(save_file)
        plotter.close()


def plot_video(seg_path, ct_file,  scale = 1, down_sample_factor = 4, crop_bounds = None,
            shift=np.array((0,0,0)).reshape(-1, 3), cam_params = None, cbar_params = None, arrow_lim = (1,5),
            clim = (1,7), show_clip_panel = False, plot_grains = False, surface_transparency = 0.05,
            velocity_split=None, show_arrowhead=False, plot_low = True, save_folder = None, interactive = True):
    """Render a per-frame animation: pore surface (from `seg_path`) plus each
    frame's particle-track velocity glyphs (from `ct_file`, a
    linked-trajectories CSV), one frame at a time.

    If `interactive`, opens an interactive pyvista window with a frame
    slider; otherwise renders off-screen and saves one screenshot per frame
    into `save_folder` (created if missing) via `Explorer3D.auto_animation`.
    """
    os.makedirs(save_folder, exist_ok=True)
    rock_surface = PoreStructure_CT(
        seg_path,  # noqa
        scale=scale,
        down_sample_factor=down_sample_factor,
        permute_axes=(2, 1, 0))

    particle_iterator_ct, num_ct_frames = load_particles_df(ct_file, shift,
                                                  arrow_lim=arrow_lim,
                                                  velocity_split=velocity_split,
                                                  show_arrowhead=show_arrowhead,
                                                  plot_low=plot_low, plot_all_tracks=False)

    plotter, explorer = setup_plotter(particle_iterator_ct, rock_surface, show_clip_panel, crop_bounds = crop_bounds,
                                clim = clim, show_labels=False, plot_grains = plot_grains,
                                surface_transparency = surface_transparency, interactive = interactive,
                                cam_params = cam_params, cbar_params = cbar_params)
    if interactive:
        plotter.show()
    elif save_folder is not None:
        explorer.auto_animation(endFrame=num_ct_frames, frameInterval=1, save_folder=save_folder)


# ---------------------------------------------------------------------------
# Raw ctracks reconstruction (`scan_*.npy`-style) particle sources, and the
# interactive/GIF-saving single- and dual-panel viewers shared by
# `data_exploration/single_view.py` and `data_exploration/duo_view.py`.
# ---------------------------------------------------------------------------

def load_particles_ctracks(particle_path_fmt, track_set='reconstruction', total_shift=np.array((0, 0, 0)).reshape(-1, 3),
                           arrow_lim=(1, 5), plot_all_tracks=False, velocity_split=None, show_arrowhead=False, plot_low=True):
    """Build one or two `ParticleIterator_CTrack` iterators (high/low velocity split) from raw
    `scan_*.npy`-style reconstruction files matched by `particle_path_fmt`.

    Returns (iterators, num_frames).
    """
    file_list = glob.glob(particle_path_fmt)
    file_list.sort(key=lambda f: int(re.search(r'(\d+)', f).group()))
    particle_high_iterator = ParticleIterator_CTrack(
        "particle_high",
        file_list,
        track_set=track_set,
        shift_array=total_shift,
        arrow_lim=arrow_lim,
        velocity_split=(velocity_split, None),
    )
    num_frames = particle_high_iterator.df[particle_high_iterator.frame_key].nunique()
    if plot_all_tracks:
        particle_high_iterator.df[particle_high_iterator.frame_key] = 0  # set all vectors to be the same frame
    particle_high_iterator.show_arrowhead = show_arrowhead

    if velocity_split is None or plot_low is False: return [particle_high_iterator], num_frames

    particle_low_iterator = ParticleIterator_CTrack(
        "particle_low",
        file_list,
        track_set=track_set,
        shift_array=total_shift,
        arrow_lim=(arrow_lim[0] / 3, arrow_lim[1] / 3),
        velocity_split=(None, velocity_split),
    )
    if plot_all_tracks:
        particle_low_iterator.df[particle_low_iterator.frame_key] = 0  # set all vectors to be the same frame
    particle_low_iterator.show_arrowhead = show_arrowhead

    return [particle_low_iterator, particle_high_iterator], num_frames


def load_particles_ctracks_dual(particle_path_fmt, total_shift=np.array((0, 0, 0)).reshape(-1, 3), arrow_lim=(1, 5),
                                plot_all_tracks=False, velocity_split=None, show_arrowhead=False, plot_low=True):
    """Build both the 'ground truth' and 'reconstruction' particle-iterator sets from the same
    `scan_*.npy`-style files matched by `particle_path_fmt` (comparing a simulated
    reconstruction against its known ground truth).

    Returns (ground_truth_iterators, reconstruction_iterators), each an (iterators, num_frames) pair.
    """
    kwargs = dict(total_shift=total_shift, arrow_lim=arrow_lim, plot_all_tracks=plot_all_tracks,
                 velocity_split=velocity_split, show_arrowhead=show_arrowhead, plot_low=plot_low)
    ground_truth = load_particles_ctracks(particle_path_fmt, track_set='ground truth', **kwargs)
    reconstruction = load_particles_ctracks(particle_path_fmt, track_set='reconstruction', **kwargs)
    return ground_truth, reconstruction


def setup_plotter_duo(particle_iterator_left, particle_iterator_right, rock_surface, show_clip_panel,
                      crop_bounds=None, clim=(1, 7), show_labels=False, plot_grains=False,
                      surface_transparency=0.05, labels=("Left", "Right"), cam_params=None):
    """Create a two-panel pyvista Plotter, build both `Explorer3D` scenes, link the views, and set the camera.

    Returns (plotter, explorer_left, explorer_right).
    """
    plotter = pv.Plotter(
        title=f"{labels[0]} vs {labels[1]}",
        shape=(1, 2),
        window_size=[2000, 1000])

    explorer_left = Explorer3D(
        velocity_iterators=particle_iterator_left,
        pore_structure=rock_surface,
        num_frames=len(particle_iterator_left[0]),
        plotter=plotter,
        clip_panel=show_clip_panel,
        clim=clim,
        crop_bounds=crop_bounds,
        plot_grains=plot_grains,
        surface_transparency=surface_transparency,
    )
    explorer_right = Explorer3D(
        velocity_iterators=particle_iterator_right,
        pore_structure=rock_surface,
        num_frames=len(particle_iterator_right[0]),
        plotter=plotter,
        clip_panel=show_clip_panel,
        clim=clim,
        crop_bounds=crop_bounds,
        plot_grains=plot_grains,
        surface_transparency=surface_transparency,
    )

    plotter.subplot(0, 0)
    plotter.show_grid(all_edges=True, show_xlabels=show_labels, show_ylabels=show_labels, show_zlabels=show_labels)
    plotter.add_text(labels[0], font_size=20)
    explorer_left.set_scene3d(0)

    plotter.subplot(0, 1)
    plotter.show_grid(all_edges=True, show_xlabels=show_labels, show_ylabels=show_labels, show_zlabels=show_labels)
    plotter.add_text(labels[1], font_size=20)
    explorer_right.set_scene3d(0)

    plotter.link_views()
    setup_cam(cam_params, plotter)

    return plotter, explorer_left, explorer_right


def plot_explorer(plotter, update_func, num_frames=20):
    """Show `plotter` interactively, with a frame slider (bound to `update_func`) if there is more than one frame."""
    if num_frames > 1:
        plotter.add_slider_widget(update_func, [0, num_frames], value=0, title='Frame')
    plotter.show()


def save_explorer(plotter, update_func, filepath="./animation.gif", move_camera=False, num_frames=20):
    """Render `num_frames` frames of `plotter` (calling `update_func` per frame) to an animated GIF at `filepath`.

    If `move_camera`, slowly orbits the camera azimuth over the animation.
    """
    plotter.open_gif(filepath, fps=1.2)
    text_actor = plotter.add_text("Frame: 0", position="upper_right", font_size=20)
    plotter.camera.zoom(1.2)
    for i in range(num_frames):
        plotter.remove_actor(text_actor)
        text_actor = plotter.add_text(f"Frame: {i}", position="upper_right", font_size=20)
        update_func(i)
        j = i / 10
        if move_camera:
            plotter.camera.azimuth = plotter.camera.azimuth - j * 2
        plotter.render()
        plotter.write_frame()

    plotter.close()


def plot_interactive(seg_path, iterators, scale=1, down_sample_factor=4, crop_bounds=None,
                     clim=(1, 7), show_clip_panel=False, save_fig=False, move_camera=False,
                     savefile="./animation.gif", plot_grains=False, surface_transparency=0.05,
                     cam_params=None):
    """Build the pore surface and show/save an interactive single-panel 3D view with a frame
    slider, scrubbing through `iterators` (an (iterators, num_frames) pair, e.g. from
    `load_particles_df` or `load_particles_ctracks`).

    Unlike `plot_single`/`plot_video` above (which render a screenshot or per-frame screenshot
    folder for headless/production use), this keeps the interactive frame-slider workflow used
    by the `data_exploration/` scripts, and can optionally save the scrubbed animation as one GIF.
    """
    rock_surface = PoreStructure_CT(
        seg_path,  # noqa
        scale=scale,
        down_sample_factor=down_sample_factor,
        permute_axes=(2, 1, 0))

    particle_iterator, num_frames = iterators

    plotter, explorer = setup_plotter(particle_iterator, rock_surface, show_clip_panel, crop_bounds=crop_bounds,
                                      clim=clim, show_labels=False, plot_grains=plot_grains,
                                      surface_transparency=surface_transparency, interactive=True,
                                      cam_params=cam_params)

    def update_view(frame_idx):
        explorer.update_scene3d(frame_idx)

    if not save_fig:
        plot_explorer(plotter, update_view, num_frames=num_frames)
    else:
        save_explorer(plotter, update_view, move_camera=move_camera, num_frames=num_frames, filepath=savefile)


def plot_duo(seg_path, left_iterators, right_iterators, scale=1, down_sample_factor=4, crop_bounds=None,
            clim=(1, 7), show_clip_panel=False, save_fig=False, move_camera=False, savefile="./animation.gif",
            plot_grains=False, surface_transparency=0.05, labels=("Left", "Right"), cam_params=None):
    """Build the pore surface and show/save a linked dual-panel 3D view comparing
    `left_iterators` and `right_iterators` (each an (iterators, num_frames) pair, e.g. from
    `load_particles_df` or `load_particles_ctracks`/`load_particles_ctracks_dual`)."""
    rock_surface = PoreStructure_CT(
        seg_path,  # noqa
        scale=scale,
        down_sample_factor=down_sample_factor,
        permute_axes=(2, 1, 0))

    particle_iterator_left, num_left = left_iterators
    particle_iterator_right, num_right = right_iterators
    num_frames = max(num_left, num_right)

    plotter, explorer_left, explorer_right = setup_plotter_duo(
        particle_iterator_left, particle_iterator_right, rock_surface, show_clip_panel,
        crop_bounds=crop_bounds, clim=clim, show_labels=True, plot_grains=plot_grains,
        surface_transparency=surface_transparency, labels=labels, cam_params=cam_params)

    def update_duo_view(frame_idx):
        plotter.subplot(0, 0)
        explorer_left.update_scene3d(frame_idx)
        plotter.subplot(0, 1)
        explorer_right.update_scene3d(frame_idx)

    if not save_fig:
        plot_explorer(plotter, update_duo_view, num_frames=num_frames)
    else:
        save_explorer(plotter, update_duo_view, move_camera=move_camera, num_frames=num_frames, filepath=savefile)

