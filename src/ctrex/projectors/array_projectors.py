"""Ray-tracing projectors for array/volume-like sample components (as opposed to particle-track
components in ``ctracks``): ``ArrayIntersectionCTSimulation`` computes analytic ray/shape
intersections directly on the projection grid, and ``RayTraceCTSimulation`` instead marches
along each ray and samples the shape model's attenuation at discrete steps.
"""
import torch
from torch.utils.checkpoint import checkpoint
import warnings

from ctrex.projectors.base_modules import CTModule


class ArrayCTSimulation(CTModule):
    """Base class for array/volume-based projectors; adds no behaviour of its own beyond
    ``CTModule``, but groups this family of projectors under a common type."""
    def __init__(self, sample_component):
        super().__init__(sample_component)


class ArrayIntersectionCTSimulation(ArrayCTSimulation):
    """Projector that computes optical depths via analytic ray/shape-model intersections on the
    full detector grid at once (see ``ray_trace``/``shape_model.optical_depths``), rather than by
    marching along rays and sampling. Used e.g. for the flow-cell and cylinder-array components."""
    def __init__(self, sample_component):
        super().__init__(sample_component)

    def add_batch(self, sinogram, sino_patches, patch_coordinates):
        """Scatter-add one batch's projection grid values (``sino_patches``) into ``sinogram`` at
        the absolute pixel coordinates given by ``patch_coordinates`` (from ``calc_vuwrite``)."""
        num_projections = len(sino_patches)
        u_ints, u_writes, v_ints, v_writes = self.calc_vuwrite(*patch_coordinates)
        batch_projection_indices = torch.arange(0, num_projections, dtype = torch.int32, device = sinogram.device)
        batch_projection_indices = batch_projection_indices.view(-1, 1, 1).expand(*sino_patches.shape)
        v_flat = v_writes.flatten()
        u_flat = u_writes.flatten()
        a_flat = batch_projection_indices.int().flatten()
        values_flat = sino_patches.flatten()
        sinogram.index_put_((v_flat, a_flat, u_flat), values_flat, accumulate=True)
        return sinogram

    @property
    def projection_grid(self):
        """ """
        if self._projection_grid is not None:
            return self._projection_grid
        device = self.ct_trajectory.device
        u_patch_1d = torch.arange(0, self.ct_trajectory.detector.rwidth, dtype = torch.int32, device=device)
        v_patch_1d = torch.arange(0, self.ct_trajectory.detector.rheight, dtype = torch.int32, device=device)
        self._projection_grid = torch.meshgrid(v_patch_1d, u_patch_1d, indexing='ij')
        return self._projection_grid

    @projection_grid.setter
    def projection_grid(self, grid):
        self._projection_grid = grid
        
    def ray_trace(self, sinogram, views, sampled_projections):
        """Evaluate the track model's world-space position(s) for ``sampled_projections``, cast
        rays for every detector pixel in ``views``, and add the shape model's analytic optical
        depths (``shape_model.optical_depths``) into ``sinogram``."""
        track_time = self.ct_trajectory.projection_time(sampled_projections)
        component_track = self.track_model(track_time)
        component_track_world = self.ct_trajectory.voxels_to_world(component_track)

        v_grid, u_grid = self.projection_grid
        u_grid, v_grid = u_grid.unsqueeze(0).unsqueeze(0), v_grid.unsqueeze(0).unsqueeze(0)
        rays = self.ct_trajectory.get_rays(u_grid, v_grid, views)  # in world coordinates
        sinogram, projection_offsets = self.shape_model.optical_depths(
            sinogram, self.projection_grid, rays, component_track_world, self.ct_trajectory.voxel_size)
        return sinogram

    def forward(self, sinogram, views, sampled_projections):
        """Add this component's analytic ray-intersection contribution into ``sinogram``."""
        sinogram = self.ray_trace(sinogram, views, sampled_projections)
        return sinogram


class RayTraceCTSimulation(ArrayIntersectionCTSimulation):
    """Marches along each ray in small steps and sums the shape model's sampled attenuation at
    every step to approximate the optical depth, batching over projections (``batch_projection_size``)
    and steps (``batch_step_size``) to bound memory use; ``sampling_rate`` controls how many steps
    are taken per voxel.

    NOTE: this class is shadowed by ``ctrex.projectors.static_projectors.RayTraceCTSimulation`` -
    ``ctrex/projectors/__init__.py`` imports this class under the same name and then immediately
    re-imports (and overwrites) it with the static_projectors version, and no other code in this
    repo imports this class directly (``from ctrex.projectors.array_projectors import
    RayTraceCTSimulation``). It therefore appears to be dead/unreachable code as currently wired up.
    """
    def __init__(self, sample_component,
                 batch_projection_size = 1, batch_step_size = 1, sampling_rate = 1):
        super().__init__(sample_component)
        self.batch_projection_size = batch_projection_size
        self.batch_step_size = batch_step_size
        self.step_size = 1
        self.sampling_rate = sampling_rate

    def ray_trace(self, sinogram, views, sampled_projections):
        """March along each ray in steps of ``step_size`` between the min/max distances bounding
        the shape model, sampling and summing attenuation at each step (batched over projections
        and steps), and add the resulting optical depth into ``sinogram`` via ``add_batch``."""
        trajectory = self.ct_trajectory
        centers_time = self.sample_component.centers_time(sampled_projections)
        projection_grid = self.projection_grid
        v_grid, u_grid = projection_grid
        origins, directions = self.ct_trajectory.get_rays(u_grid, v_grid, views)  # in world coordinates
        lengths = torch.linalg.norm(directions, dim = -1, keepdim = True)
        directions /= lengths
        min_distance = trajectory.sod / trajectory.voxel_size - self.sample_component.shape_model.shape[1] / 2
        max_distance = trajectory.sod / trajectory.voxel_size + self.sample_component.shape_model.shape[1] / 2
        step_size = self.sample_component.shape.voxel_scale / self.sampling_rate
        steps = torch.arange(min_distance, max_distance, step_size, device = sinogram.device)
        bps = self.batch_projection_size
        for pi in range(0, sinogram.shape[1], bps):  # per projection subbatch.
            projection_offsets = torch.zeros(2, 1, len(origins[0, pi:pi+bps]), dtype = torch.int32,
                                             device = sinogram.device)  # 1 particle
            for si in range(0, len(steps), self.batch_step_size):
                batch_steps = steps[si: si + self.batch_step_size].view(-1, 1, 1, 1, 1)
                voxel_positions = (origins[0, pi:pi+bps].view(1, -1, 1, 1, 3)
                                   + batch_steps * trajectory.voxel_size * directions[0, pi:pi+bps])
                voxel_positions = trajectory.world_to_voxels(voxel_positions)
                voxel_positions = voxel_positions.reshape((len(batch_steps) * len(origins[0, pi:pi+bps]),)
                                                          + voxel_positions.shape[2:])
                attenuations = self.sample_component.shape_model.sample_attenuation(voxel_positions, centers_time[:, pi:pi + bps],
                                                                                    self.ct_trajectory.voxel_size, sampled_projections)
                attenuations = attenuations.reshape((len(batch_steps), len(origins[0, pi: pi+bps]))
                                                    + attenuations.shape[1:])
                attenuations = attenuations.sum(dim = 0)  # sum over steps
                # divide by 10 since voxel size is in mm and attenuation is in 1/cm
                optical_depth = trajectory.voxel_size * step_size / 10 * attenuations
                sinogram[:,pi:pi+bps] = self.add_batch(sinogram[:,pi:pi+bps], optical_depth,
                                                          (*projection_offsets, *projection_grid))
        return sinogram

    def add_batch(self, sinogram, sino_patches, patch_coordinates):
        """Like ``ArrayIntersectionCTSimulation.add_batch``, but runs the scatter-add under
        ``torch.utils.checkpoint`` (recomputing it on the backward pass instead of storing its
        intermediates) and works on a cloned sinogram to avoid in-place modification."""
        # Create temporary copy of the batch to avoid inplace modification of the sinogram
        warnings.filterwarnings("ignore", category=UserWarning, message="The .grad attribute of a Tensor.*")
        temp_sinogram = torch.zeros_like(sinogram)
        # checkpoint tells PyTorch: Don’t store the intermediate tensors from ray_trace_batch
        # Instead, re-run the forward pass during backward() to get the gradients.
        temp_sinogram = checkpoint(super().add_batch,temp_sinogram, sino_patches, patch_coordinates,
                                   use_reentrant=True)
        temp_sinogram = temp_sinogram + sinogram
        sinogram = torch.clone(temp_sinogram)
        if temp_sinogram.requires_grad:
            sinogram.retain_grad()
        return sinogram

    def forward(self, sinogram, views, sampled_projections):
        """Add this component's step-marched ray-trace contribution into ``sinogram``."""
        sinogram = self.ray_trace(sinogram, views, sampled_projections)
        return sinogram
