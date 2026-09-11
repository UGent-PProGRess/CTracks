"""
Author Chunyang Wang
Github: https://github.com/chunyang-w

A Tool that utilise pyvista API to provide a interactive 3D scene
for users to explore the surface of flow in porous media.

2D slicers are also provided to allow users to explore the internal
structure of the porous media.

If information about velocity field is provided, the class will also
plot the velocity field within the surface of the porous media.

Users can interact with the 3D scene by rotating, zooming, and panning,
as well as using a slider bar to move along time axis.

--------------------------------------------------------------------------
Role in the data flow
--------------------------------------------------------------------------
`Explorer3D` is the scene-assembly step (see `plot_3d_vectors.py` for the full
pipeline: mesh generation -> particle iteration -> scene assembly -> render/save).
It does not compute any geometry itself - it is handed already-built providers
(a `PoreStructure` for the static pore/mask mesh, and/or one or more
`ParticleIterator`s for the per-frame velocity-glyph meshes; see `PoreStructure.py`
and `Particle.py`) and is responsible for:
- `set_scene3d(frame_idx)`: build the pyvista meshes/actors for one frame and add
  them to `self.plotter`, remembering them (`self.fluid_surfaces`/`self.velocity_arrows`)
  so they can be updated in place later.
- `update_scene3d(frame_idx)`: swap the remembered meshes/actors to a different frame's
  data, without rebuilding the plotter from scratch - this is what makes per-frame
  animation (`auto_animation`) and the interactive frame slider (`set_time_slider`)
  cheap.
- `explore()` / `auto_animation()`: the actual render/save step - either open the
  interactive window, or step through frames headlessly and screenshot each one.
"""
import os

import numpy as np
import pyvista as pv


class Explorer3D:
    def __init__(
        self,
        fluid_iterators=None,
        velocity_iterators=None,
        pore_structure=None,
        num_frames=100,
        bg_color="white",
        surface_transparency=0.05,
        particle_cmap="jet",
        plotter=None,
        clip_panel=True,
        clim=(0, 7),
        crop_bounds = None,
        plot_grains = True,
        cbar_params = None
    ):
        """Build the pyvista scene state (no rendering happens until `set_scene3d`/`explore` are called).

        Parameters
        ----------
        fluid_iterators : list or None
            Per-frame fluid-surface mesh providers (indexable by frame), if plotting a fluid surface.
        velocity_iterators : list or None
            Per-frame particle-velocity glyph providers (see `Particle.ParticleIterator`), if plotting velocity arrows.
        pore_structure : PoreStructure or None
            Static pore/mask geometry provider (see `PoreStructure.PoreStructure`).
        num_frames : int
            Total number of frames, used by `set_time_slider`.
        plotter : pyvista.Plotter or None
            Existing plotter to draw into (e.g. one subplot of a multi-panel figure); a new one is created if None.
        clip_panel : bool
            Whether to add an interactive clip-plane widget for the fluid/pore meshes.
        clim : tuple
            Color limits for the velocity-magnitude scalar bar.
        crop_bounds : sequence or None
            [xmin, xmax, ymin, ymax, zmin, zmax] box to clip all meshes to, if given.
        plot_grains : bool
            Whether to render the pore-structure mesh at all.
        cbar_params : dict or None
            Overrides for the velocity scalar-bar appearance; see default below.
        """
        self.pore_structure = pore_structure
        self.fluid_iterators = fluid_iterators
        self.velocity_iterators = velocity_iterators
        self.num_frames = num_frames
        self.surface_transparency = surface_transparency
        self.particle_cmap = particle_cmap
        self.plotter = plotter
        self.clip_panel = clip_panel
        self.clim = clim
        self.crop_bounds = crop_bounds
        self.plot_grains = plot_grains

        self.cbar_params = cbar_params
        if self.cbar_params is None:
            self.cbar_params = dict(title="Velocity magnitude\n(vox/scan)", height=0.35, vertical=True,
                         position_x=0.05, position_y=0.35, n_labels=3, fmt="%.0f",
                         title_font_size=26, label_font_size=26, font_family='arial')

        self.setup(bg_color)
        self.set_light()

        self.fluid_surfaces = []
        self.velocity_arrows = []

    def setup(self, bg_color):
        """Create a default plotter if none was supplied, set the background, and bind the "p" camera-info key event."""
        if self.plotter is None:
            self.plotter = pv.Plotter(
                window_size=[1600, 1600],
                title="Particle-vtools (3D Explorer)")
        pv.global_theme.background = bg_color
        self.plotter.add_key_event("p", self.my_cpos_callback)

    def set_light(self):
        """Configure a directional light for the scene.

        NOTE: the constructed `light` is never attached to `self.plotter`
        (no `add_light` call) - possibly incomplete/dead code left as-is
        since fixing it would change the rendered lighting; flagged here.
        """
        light = pv.Light()
        light.set_direction_angle(30, 30)

    def set_scene3d(self, frame_idx):
        """Build and add all requested meshes (fluid surfaces, velocity glyphs, pore structure) for `frame_idx`."""
        frame_idx = int(frame_idx)
        print("Setting scene to frame", frame_idx)
        # set fluid surface
        if self.fluid_iterators is not None:
            for fluid_iterator in self.fluid_iterators:
                fluid_mesh = fluid_iterator[frame_idx]
                if self.crop_bounds is not None:
                    fluid_mesh = fluid_mesh.clip_box(bounds = self.crop_bounds, invert=False, crinkle=True)
                self.fluid_surfaces.append(fluid_mesh)
                self.plotter.add_mesh(
                    fluid_mesh,
                    color="blue",
                    pbr=True,
                    metallic=0.1,
                    roughness=0.01,
                    diffuse=1,
                    opacity=self.surface_transparency)
                if self.clip_panel:
                    self.plotter.add_mesh_clip_plane(
                        fluid_mesh,
                        normal='-z',
                        origin=fluid_mesh.center,
                        color="blue", outline_opacity=0.1)

        # set particle velocity arrow
        if self.velocity_iterators is not None:
            for velocity_iterator in self.velocity_iterators:
                velocity = velocity_iterator[frame_idx]
                if self.crop_bounds is not None:
                    velocity = velocity.clip_box(bounds = self.crop_bounds, invert=False)

                actor = self.plotter.add_mesh(
                    velocity,
                    cmap=self.particle_cmap,
                    clim=self.clim,
                    scalar_bar_args={**self.cbar_params},
                )
                self.velocity_arrows.append(actor)

        # set pore structure
        if self.pore_structure is not None:
            pore_mesh = self.pore_structure.get_surface()
            if self.crop_bounds is not None:
                pore_mesh = pore_mesh.clip_box(bounds=self.crop_bounds, invert=False)
            if self.plot_grains:
                self.plotter.add_mesh(
                    pore_mesh,
                    color="grey",
                    pbr=True,
                    metallic=0.1,
                    roughness=0.01,
                    diffuse=1,
                    opacity=0.1)
                if self.clip_panel:
                    self.plotter.add_mesh_clip_plane(
                        pore_mesh,
                        normal='x', origin=pore_mesh.center,
                        color="grey")
        # self.plotter.reset_camera()

    def update_scene3d(self, frame_idx):
        """Update the already-built meshes/actors in place to show `frame_idx` (used as the frame-slider callback)."""
        frame_idx = int(frame_idx)
        print(f"Updating scene to frame {frame_idx}")

        if self.fluid_iterators is not None:
            for i, fluid_iterator in enumerate(self.fluid_iterators):
                fluid_surface_i = fluid_iterator[frame_idx]
                if self.crop_bounds is not None:
                    fluid_surface_i = fluid_surface_i.clip_box(bounds = self.crop_bounds, invert=False)
                self.fluid_surfaces[i].points = fluid_surface_i.points
                self.fluid_surfaces[i].faces = fluid_surface_i.faces

        if self.velocity_iterators is not None:
            for i, velocity_iterator in enumerate(self.velocity_iterators):
                velocity_arrow_i = velocity_iterator[frame_idx]

                if self.velocity_arrows[i] is not None:
                    self.plotter.remove_actor(self.velocity_arrows[i])

                if self.crop_bounds is not None:
                    velocity_arrow_i = velocity_arrow_i.clip_box(bounds=self.crop_bounds, invert=False)

                # Use 'name' to overwrite the mesh
                self.velocity_arrows[i] = self.plotter.add_mesh(
                    velocity_arrow_i,
                    cmap=self.particle_cmap,
                    clim=self.clim,
                    name=f"velocity_vectors_{i}",
                    reset_camera=False,
                    show_scalar_bar=False
                )


        self.plotter.render()


    def auto_animation(self, startFrame=0, endFrame=100, frameInterval=1, save_folder="./"):
        """Step through frames [startFrame, endFrame) by frameInterval, updating the scene and saving one screenshot per step to `save_folder`."""
        for i in np.arange(startFrame, endFrame, frameInterval):
            self.update_scene3d(i)
            self.plotter.screenshot(filename=os.path.join(save_folder, 'frame_' + str(i).zfill(4) + '.png'))
            print("Step " + str(i) + " finished.")
        self.plotter.close()
        print("Time step images saved")

    def set_time_slider(self, start=0):
        """Add an interactive frame slider (range [start, start + num_frames - 1]) that calls `update_scene3d`."""
        end = start + self.num_frames - 1
        self.plotter.add_slider_widget(
            self.update_scene3d,
            [start, end],
            title='Frame', value=0)

    def explore(self):
        """Open the interactive pyvista window (blocking)."""
        self.plotter.show()

    def my_cpos_callback(self):
        """Key-event callback (bound to "p"): print and overlay the current camera position/orientation for picking view parameters."""
        self.plotter.add_text(str(self.plotter.camera.position), position = (100,100), name="cpos")
        print("Camera position vector: " + str(self.plotter.camera_position))
        print("Azimuth: " + str(self.plotter.camera.azimuth))
        print("Roll: " + str(self.plotter.camera.roll))
        print("Elevation: " + str(self.plotter.camera.elevation))
        print("View angle: " + str(self.plotter.camera.view_angle))
        print("Zoom: " + str(self.plotter.camera.zoom))
        return