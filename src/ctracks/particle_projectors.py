import torch
from ctrex.projectors.base_modules import CTModule


class ParticleCTSimulation(CTModule):
    """Differentiable forward projector for particles represented as spheres with a learnable 3D track.

    Projects each particle's track/shape model into small local patches per projection angle
    (see ``generate_patches``) and scatter-adds them into the sinogram (``sum_patches``). Particles
    or patches that fall outside the sinogram bounds are silently dropped.
    """

    def __init__(self, sample_component,
                 patching_batch_size = 1500, sum_batch_size = 500, patch_interpolation ='bilinear'):
        super().__init__(sample_component)
        self.patching_batch_size = patching_batch_size
        self.sum_batch_size = sum_batch_size
        self.patch_interpolation = patch_interpolation
        self.projection_grid = None
        self.patch_margin = 0

    def set_ct_trajectory(self, ct_trajectory):
        super().set_ct_trajectory(ct_trajectory)

    @property
    def projection_grid(self):
        """ Small projection grid around small particles """
        if self._projection_grid is not None:
            return self._projection_grid  # Should be equal over all runs for trace
        device = self.ct_trajectory.device

        # quick, inaccurate estimate of projections that could be somewhat orthogonal
        ortho_projections = torch.tensor([0,
                                          self.ct_trajectory.num_projections // 3,
                                          2 * self.ct_trajectory.num_projections // 3], device = device, dtype = torch.int32)
        ortho_times = self.ct_trajectory.projection_time(ortho_projections)
        component_track = self.track_model(ortho_times)
        ortho_views = self.ct_trajectory(ortho_projections)
        mag_components = self.ct_trajectory.project_voxels(component_track, ortho_views)[2]
        projected_sizes = (mag_components * self.shape_model.max_size().unsqueeze(1)).unsqueeze(-1).unsqueeze(-1)
        # projected_sizes shape: (N_particles, N_angles, 1, 1)
        max_size = torch.ceil(projected_sizes.max())  # half window size of patch in number of pixels
        patch_rad = max_size.int().item() + self.patch_margin
        u_patch_1d = torch.arange(-patch_rad, patch_rad + 1, dtype = torch.float32, device=device)
        v_patch_1d = (torch.arange(-patch_rad, patch_rad + 1, dtype = torch.float32, device=device) if self.shape_model.dims > 2
                      else torch.zeros(1, dtype = u_patch_1d.dtype, device=device))
        # noinspection PyUnusedLocal
        self._projection_grid = v_grid, u_grid = torch.meshgrid(v_patch_1d, u_patch_1d, indexing='ij')
        return self._projection_grid

    @projection_grid.setter
    def projection_grid(self, grid):
        self._projection_grid = grid

    def generate_patches(self, sampled_projections, views):
        """Evaluate the track model at the sampled projections, project particle centers into
        detector coordinates, and compute each particle's analytic sphere chord-length patch.

        Returns:
            sino_patches: per-particle, per-angle intensity patches.
            patch_coordinates: (v_particles, u_particles, v_grid, u_grid) needed to place the
                patches back into the sinogram.
        """
        projection_times = self.ct_trajectory.projection_time(sampled_projections)
        centers_time = self.track_model(projection_times)
        projected_centres = self.ct_trajectory.project_voxels(centers_time, views)
        projection_grid = self.projection_grid
        sino_patch = projection_grid[0]*0
        sino_patches, projection_offsets = self.shape_model.optical_depths(sino_patch, projection_grid, projected_centres)

        return sino_patches, (*projection_offsets[:2], *projection_grid[:2])[:4]

    def align_patch(self, sino_patches, u_ints, u_particles, u_grid, v_ints, v_particles, v_grid):
        """Sub-pixel-align each particle's patch to its true (non-integer) projected position via
        ``grid_sample``, preserving differentiability w.r.t. sub-pixel position."""
        num_particles = u_particles.shape[0]
        num_angles = sino_patches.shape[1]
        v_patch_len, u_patch_len = v_grid.shape
    
        template_grid = torch.stack((v_grid, u_grid), dim=-1)  # [v_len, u_len, 2]
        template_grid = template_grid.unsqueeze(0).unsqueeze(0).expand(num_particles, num_angles, -1, -1, -1).float()
    
        v_offsets = (v_ints - v_particles).view(num_particles, num_angles, 1, 1, 1
                                                ).expand(-1, -1, v_patch_len, u_patch_len, 1).float()
        u_offsets = (u_ints - u_particles).view(num_particles, num_angles, 1, 1, 1
                                                ).expand(-1, -1, v_patch_len, u_patch_len, 1).float()
        offset_grid = torch.cat((v_offsets, u_offsets), dim=-1)  # xy order
    
        align_grid = (template_grid + offset_grid)
    
        # Normalize to [-1, 1]
        align_grid[..., 0] /= (v_patch_len / 2)  # ((patch_rad - (patch_rad - 1)) - 1) / 2 = patch_rad
        align_grid[..., 1] /= (u_patch_len / 2)
    
        # Reshape for grid_sample: [N, H, W, 2]
        align_grid = align_grid.view(num_particles * num_angles, v_patch_len, u_patch_len, 2)
    
        # Reshape sino_patches to match: [N, Channels=1, H, W]
        sino_patches = sino_patches.view(num_particles * num_angles, 1, sino_patches.size(2), sino_patches.size(3))
    
        # Sample
        aligned_patches = torch.nn.functional.grid_sample(sino_patches, align_grid, align_corners=True,
                                                          mode=self.patch_interpolation).squeeze(1)
        aligned_patches = aligned_patches.reshape(num_particles, num_angles, v_patch_len, u_patch_len)
    
        return aligned_patches

    def sum_patches(self, sinogram, sino_patches, patch_coordinates, clamp = False, check_bounds = True):
        """Scatter-add ``sino_patches`` into ``sinogram``, batched over particles (``sum_batch_size``).

        Args:
            clamp: if True, clamp particle coordinates into bounds instead of masking them out
                (used by ``ManyParticleCTSimulation``).
            check_bounds: if True, drop particles/patches that fall outside the sinogram.
        """
        v_particles, u_particles, v_grid, u_grid = patch_coordinates
        batch_size = self.sum_batch_size
        num_particles = u_particles.shape[0]

        for i in range(0, num_particles, batch_size):  # batching addition to sinogram over particles
            batch_u_particles = u_particles[i:i + batch_size]
            batch_v_particles = v_particles[i:i + batch_size]

            # TODO clamp per batch or clamp all at once? - per batch seems faster?
            if clamp:
                batch_u_particles.clamp_(min=sino_patches.shape[2] // 2,
                                         max=sinogram.shape[2] - sino_patches.shape[2] // 2 - 1)
                batch_v_particles.clamp_(min=sino_patches.shape[3] // 2,
                                         max=sinogram.shape[0] - sino_patches.shape[3] // 2 - 1)

            batch_sino_patches = sino_patches[i:i + batch_size]

            self.add_batch(sinogram, batch_sino_patches, (batch_v_particles, batch_u_particles, v_grid, u_grid),
                           check_bounds=check_bounds)
        return sinogram

    def add_batch(self, sinogram, sino_patches, patch_coordinates, check_bounds = True):
        """Align one batch of patches and accumulate them into ``sinogram`` via ``index_put_``,
        masking out any (particle, angle, pixel) combination that falls outside the sinogram."""
        in_bounds = slice(None)
        v_particles, u_particles, v_grid, u_grid = patch_coordinates
        u_ints, u_writes, v_ints, v_writes = self.calc_vuwrite(*patch_coordinates)
        if check_bounds:
            in_bounds = torch.where(((0 <= v_writes) & (v_writes < sinogram.shape[0])
                                     & (0 <= u_writes) & (u_writes < sinogram.shape[2])))
        aligned_patches = self.align_patch(sino_patches, u_ints, u_particles, u_grid, v_ints, v_particles, v_grid)
        num_particles, num_projections = u_particles.shape
        batch_projection_indices = torch.arange(0, num_projections, dtype = torch.int32, device = sinogram.device)
        batch_projection_indices = (batch_projection_indices.view(1, -1, 1, 1)
                                    .expand(num_particles,num_projections, sino_patches.shape[2],sino_patches.shape[3]))
        v_flat = v_writes[in_bounds].flatten()
        u_flat = u_writes[in_bounds].flatten()
        a_flat = batch_projection_indices[in_bounds].int().flatten()
        values_flat = aligned_patches[in_bounds].flatten()
        sinogram.index_put_((v_flat, a_flat, u_flat), values_flat, accumulate=True)

    def forward(self, sinogram, views, sampled_projections):
        """Project all particles for ``sampled_projections`` and add them into ``sinogram``."""
        sino_patches, patch_coordinates = self.generate_patches(sampled_projections, views)
        self.sum_patches(sinogram, sino_patches, patch_coordinates, clamp = False)
        return sinogram


class ManyParticleCTSimulation(ParticleCTSimulation):
    """
    Rather than checking bounds to ignore write operations,
    this class expands the sinogram size and clamps the particle coordinates to always be in bounds
    The patches are also not calculated for all particles simultaneously, but batched.
    The ParticleProjector only uses batching for the sinogram addition operation.

    NOTE: not currently instantiated anywhere in the pipeline and not verified to run end-to-end.
    Restored and realigned with the current ``ParticleCTSimulation`` constructor/``projection_grid``
    API (both had drifted out of sync with this class) on request, but still untested - keep that
    in mind before relying on it.
    """
    def __init__(self, sample_component, patching_batch_size = 1500, sum_batch_size = 500, patch_interpolation ='bilinear'):
        super().__init__(sample_component, patching_batch_size, sum_batch_size, patch_interpolation)
        self.patch_margin = 1

    @staticmethod
    def expand_sinogram(sinogram, sino_patch_width):
        """Pad ``sinogram`` by ``sino_patch_width`` on all spatial sides so that clamped particle
        coordinates always stay in bounds; returns the padded sinogram and the padding tuple used."""
        u_patch_width = sino_patch_width
        v_patch_width = sino_patch_width
        padding_tuple = (v_patch_width, v_patch_width,
                         0, 0,
                         u_patch_width, u_patch_width)

        sinogram = torch.nn.functional.pad(sinogram, padding_tuple)
        return sinogram, padding_tuple

    def forward(self, sinogram, views, sampled_projections):
        """Project all particles, expanding and clamping into a padded sinogram instead of bounds-masking."""
        # The patch size is determined based on an estimate of orthogonal projections
        projection_grid = self.projection_grid

        # rather than bounds checking, expanding sinogram and clamping particle coords into bounds to be cropped
        sino_patch_width = projection_grid[0].shape[-1]  # some safety margin of 2 pixels in both directions
        sinogram, padding_tuple = self.expand_sinogram(sinogram, sino_patch_width + 2)
        patch_rad = sino_patch_width // 2
        # make patches with the predetermined patch size
        sino_patches, patch_coordinates = self.generate_patches(sampled_projections, views)
        v_particles, u_particles, v_grid, u_grid = patch_coordinates            # add w to coords where necessary
        u_particles = u_particles + padding_tuple[0]  # TODO check this is the right order v/u
        v_particles = v_particles + padding_tuple[4]

        for i in range(0, self.track_model.num_particles, self.patching_batch_size):  # batching patch generation over particles
            batch_patch_coordinates = (v_particles[i:i + self.patching_batch_size],
                                       u_particles[i:i + self.patching_batch_size],
                                       v_grid,
                                       u_grid)
            self.sum_patches(sinogram, sino_patches[i:i + self.patching_batch_size], batch_patch_coordinates,
                             clamp = True, check_bounds=False)

        # crop sinogram to original size - removes elements that would have been bounds checked
        sinogram = sinogram[padding_tuple[4]: -padding_tuple[5], :, padding_tuple[0]: -padding_tuple[1]]
        return sinogram
