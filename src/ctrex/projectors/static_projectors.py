"""CUDA-accelerated static (non-particle) CT projectors: ray-tracing forward/back-projection and
SIRT-style reconstruction, implemented as custom ``torch.autograd.Function``s (``RayTraceProjector``,
``SIRTProjector``) that call into hand-written CUDA kernels via CuPy, with the analytic geometry
gradient computed either numerically (finite differences, ``numeric_geom_gradient``) or via a
dedicated backprop kernel (``kernel_geom_gradient``). ``RayTraceCTSimulation``/``SIRTCTSimulation``
wrap these as regular CTModules. ``check_geom_grad`` at the bottom is a standalone script for
manually checking the geometry gradient.
"""
import os
import cupy as cp
import torch
from torch import nn
from torch.utils import dlpack
from cupy.cuda import runtime

from ctrex.projectors import ArrayCTSimulation
from ctrex.utils_gpu.base_kernel import sdk_path, CudaKernel

projector_path = os.path.dirname(__file__)


class ProjectorTypeKernel(CudaKernel):
    """Texture setup shared by forward-projection-style kernels (ray tracing, ray weighting,
    geometry backprop): border addressing in u/v and clamped addressing along the ray, with
    linear (trilinear) interpolation."""
    texture_modes = ((runtime.cudaAddressModeBorder, runtime.cudaAddressModeBorder, runtime.cudaAddressModeClamp),
                     runtime.cudaFilterModeLinear,
                     runtime.cudaReadModeElementType)
    module_options = CudaKernel.module_options + (f'-I {projector_path}',)

class BackProjectorTypeKernel(CudaKernel):
    """Texture setup shared by backprojection-style kernels (backprojection, voxel weighting):
    border addressing on all three axes, with linear (trilinear) interpolation."""
    texture_modes = ((runtime.cudaAddressModeBorder,) * 3,
                     runtime.cudaFilterModeLinear,
                     runtime.cudaReadModeElementType)
    module_options = CudaKernel.module_options + (f'-I {projector_path}',)

class ProjectorKernel(ProjectorTypeKernel):
    """Loads the ``static_projection.cu`` forward-projection (ray-tracing) kernel."""
    def __init__(self, kernel_path=os.path.join(projector_path, "static_projection.cu")):
        super().__init__('static_projection', kernel_path)

class BackprojectorKernel(BackProjectorTypeKernel):
    """Loads the ``static_backprojection.cu`` backprojection kernel."""
    def __init__(self, kernel_path=os.path.join(projector_path, "static_backprojection.cu")):
        super().__init__('static_backprojection', kernel_path)


class RayweightingKernel(ProjectorTypeKernel):
    """Loads the ``ray_weighting.cu`` kernel that applies the SIRT-style ray (row) preconditioning
    to the sinogram residual before backprojection."""
    def __init__(self, kernel_path=os.path.join(projector_path, "ray_weighting.cu")):
        super().__init__('ray_weighting', kernel_path)


class VoxelweightingKernel(BackProjectorTypeKernel):
    """Loads the ``voxel_weighting.cu`` kernel that applies the SIRT-style voxel (column)
    preconditioning to the backprojected volume gradient."""
    def __init__(self, kernel_path=os.path.join(projector_path, "voxel_weighting.cu")):
        super().__init__('voxel_weighting', kernel_path)


class GeomBackpropKernel(ProjectorTypeKernel):
    """Loads the ``geom_backprop.cu`` kernel that analytically backpropagates the sinogram
    gradient into the per-view geometry matrix (an alternative to the numerical finite-difference
    gradient in ``RayTraceProjector.numeric_geom_gradient``). Only exercised by the standalone
    ``check_geom_grad`` debug utility at the bottom of this module."""
    def __init__(self, kernel_path=os.path.join(projector_path, "geom_backprop.cu")):
        super().__init__('geom_backprop', kernel_path)


class RayTraceProjector(torch.autograd.Function):
    """Custom autograd Function implementing differentiable CUDA ray-trace forward projection and
    backprojection of a static volume. ``forward`` runs the ray-tracing kernel to produce a
    sinogram from ``volume``; ``backward`` runs the backprojection kernel to get the volume
    gradient, plus (if the view/geometry matrices require gradients) a numerical finite-difference
    gradient through the geometry (``numeric_geom_gradient``). ``args_fp``/``args_bp`` build the
    positional CUDA-kernel argument tuples shared by forward/backward and by ``SIRTProjector``."""

    @staticmethod
    def args_fp(ctx, volume, sinogram, centers_time, projection_indices, voxel_scale, trajectory, views, sampling_rate,
                extend_top = False, extend_bottom = False):
        """Build the positional CuPy-kernel argument tuple for the forward ray-trace kernel:
        computes the volume-of-interest centre/bounding box around ``centers_time`` and packs it
        together with ``views``, ``projection_indices``, and detector/geometry sizes. Also stashes
        the values needed later on ``ctx`` for ``backward`` (voxel/pixel/step sizes, trajectory,
        ROI offsets, whether ``views`` requires grad)."""
        sino_shape = sinogram.shape
        voxel_size = float(trajectory.voxel_size)
        pixel_size = float(trajectory.pixel_size)
        step_size = voxel_size * voxel_scale / sampling_rate
        roi_u_start, roi_v_start = trajectory.detector.roi[1].start, trajectory.detector.roi[0].start

        # find voi centre
        voi_shape = cp.array(volume.shape[::-1])  # depth, height, width
        vol_shape = cp.array(trajectory.volume.shape[::-1])  # detector_width, detector_width, detector_height
        voi_centres = 0.5 * voi_shape * voxel_scale + (0.5 * vol_shape * 1 - centers_time)  # static alignment, determined by external registration
        chunk_box_rad = (voi_shape - 2) * 0.5 * voxel_scale  # -2: stay completely within volume all sides
        box_min = (- chunk_box_rad + (0.5 * vol_shape - centers_time[0,0])) * voxel_size
        box_max = (chunk_box_rad + (0.5 * vol_shape - centers_time[0,0])) * voxel_size
        if extend_bottom:
            box_min[2] *= 2
        if extend_top:
            box_max[2] *= 2

        # depth, height, width = volume.shape
        # min_distance, max_distance = trajectory.distances_entry_exit(volume, centers_time)
        num_projections_subset, num_subviews = views.shape[:2]
        num_rows, num_projections_subset, num_cols = sino_shape
        # zero_sized shapes will cause illegal memory reads
        assert not any((0 in arg.shape) for arg in (centers_time, views, projection_indices))

        args_fp = (cp.asarray(views.contiguous().detach().to('cuda'), dtype=cp.float32),
                   cp.asarray(voi_centres, dtype = cp.float32),
                   cp.asarray(box_min, dtype = cp.float32),
                   cp.asarray(box_max, dtype = cp.float32),
                   # cp.asarray(origins.detach())
                   # cp.asarray((directions / torch.linalg.norm(directions, dim=-1, keepdim=True)).detach())
                   cp.asarray(projection_indices, dtype = cp.uint32),
                   # cp.uint32(width),
                   # cp.uint32(height),
                   # cp.uint32(depth),
                   cp.uint32(num_subviews),
                   cp.uint32(num_rows),
                   cp.uint32(num_projections_subset),
                   cp.uint32(num_cols),
                   cp.uint32(trajectory.detector.height),
                   cp.uint32(trajectory.detector.width),
                   cp.float32(pixel_size),
                   # cp.float32(min_distance),
                   cp.float32(step_size),
                   # cp.float32(max_distance),
                   cp.float32(voxel_scale),
                   cp.float32(voxel_size),
                   cp.uint32(roi_u_start),
                   cp.uint32(roi_v_start))

        # Save values for backward pass
        ctx.voxel_scale = voxel_scale
        ctx.voxel_size = voxel_size
        ctx.pixel_size = pixel_size
        ctx.step_size = step_size
        ctx.trajectory = trajectory
        ctx.roi_u_start = roi_u_start
        ctx.roi_v_start = roi_v_start
        ctx.views_requires_grad = views.requires_grad
        return args_fp

    @staticmethod
    def args_bp(ctx, volume_cp, sinogram, centers_time, projection_indices, views):
        """Build the positional CuPy-kernel argument tuple for the backprojection kernel, mirroring
        ``args_fp`` but reading the saved geometry/scale values back off ``ctx`` instead of
        receiving them as arguments."""
        trajectory = ctx.trajectory
        voxel_scale = ctx.voxel_scale
        voxel_size = ctx.voxel_size
        pixel_size = ctx.pixel_size
        step_size = ctx.step_size
        detector_width, detector_height = trajectory.detector.width, trajectory.detector.height
        detector_rheight, num_projections_subset, detector_rwidth = sinogram.shape
        roi_u_start, roi_v_start = ctx.roi_u_start, ctx.roi_v_start
        depth, height, width = volume_cp.shape
        # find voi centre
        voi_shape = cp.array(volume_cp.shape[::-1])  # depth, height, width
        vol_shape = cp.array(trajectory.volume.shape[::-1])  # detector_width, detector_width, detector_height
        voi_centres = 0.5 * voi_shape * voxel_scale + (
                    0.5 * vol_shape * 1 - centers_time)  # static alignment, determined by external registration

        args_bp = (cp.asarray(views.detach().to('cuda'), dtype=cp.float32),
                   cp.asarray(voi_centres, dtype=cp.float32),
                   cp.asarray(projection_indices, dtype = cp.uint32),
                   cp.uint32(width),
                   cp.uint32(height),
                   cp.uint32(depth),
                   cp.float32(voxel_scale),
                   cp.float32(voxel_size),
                   cp.float32(pixel_size),
                   cp.float32(step_size),
                   cp.uint32(num_projections_subset),
                   cp.uint32(detector_width),
                   cp.uint32(detector_height),
                   cp.uint32(roi_u_start),
                   cp.uint32(roi_v_start),
                   cp.uint32(detector_rwidth),
                   cp.uint32(detector_rheight))

        return args_bp

    @staticmethod
    def numeric_geom_gradient(projector: ProjectorKernel, grad_sino, views_requires_grad, device, input_data, sino_shape,
                              args_fp, eps = 1e-3, bounds = 1):
        """Estimate the gradient of the sinogram w.r.t. each nonzero component of the per-view
        geometry matrix by finite differences: perturb each component by ``eps``, re-run the
        forward projection kernel, and divide the resulting sinogram change (weighted by
        ``grad_sino``, summed over u/v with ``bounds`` border pixels clipped) by ``eps``. Returns
        None if ``views_requires_grad`` is False (and not tracing), since the gradient is then
        unused."""
        if not views_requires_grad and not torch.jit.is_tracing():
            grad_views = None
            return grad_views
        # Numerical geometry gradient
        views_cp = args_fp[0]
        grad_views = torch.zeros(views_cp.shape, device = device, dtype = torch.float32)
        sinogram = cp.zeros(sino_shape, dtype=cp.float32)
        projector(*input_data, sinogram, views_cp, *args_fp[1:])
        for view_ij in (0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14):  # Nonzero components
            view_i, view_j = view_ij // 4, view_ij % 4
            views_delta_cp = views_cp.copy()
            views_delta_cp[..., view_i, view_j] += eps
            sinogram_delta = cp.zeros(sino_shape, dtype=cp.float32)
            projector(*input_data, sinogram_delta, views_delta_cp, *args_fp[1:])
            grad_sino_num = grad_sino * (sinogram_delta - sinogram) / eps
            # Clip bounds
            grad_sino_num = cp.sum(grad_sino_num[..., bounds:sino_shape[2] - bounds], axis=-1, keepdims=False)  # over u, clip borders
            grad_sino_num = cp.sum(grad_sino_num[bounds:sino_shape[0] - bounds], axis=0, keepdims=False)  # over v -> shape (Np, 1)
            grad_views[:, 0, view_i, view_j] = torch.as_tensor(grad_sino_num, dtype = grad_views.dtype,
                                                               device = grad_views.device)  # 0 subviews

        return grad_views

    @staticmethod
    def kernel_geom_gradient(grad_output, views_cp, volume_cp, args_fp, geombackpropagator):
        """Analytic alternative to ``numeric_geom_gradient``: runs ``geombackpropagator``
        (``GeomBackpropKernel``) to backpropagate the sinogram gradient directly into the 16
        components of the per-view geometry matrix, instead of estimating it by finite
        differences.

        NOTE: not currently used/called anywhere in this repo - `check_geom_grad` at the bottom of
        this module exercises `GeomBackpropKernel` directly rather than through this method, and
        nothing else calls it either.
        """
        # Analytical geometry gradient
        grad_sinogram_cp = cp.asarray(grad_output[0].to(torch.float32).detach().contiguous(), dtype=cp.float32)
        grad_view_sino = cp.tile(grad_sinogram_cp[...,cp.newaxis] * 0, (1, 1, 1, 16))  # (Nv, Np, Nu, 16)
        geombackpropagator(grad_sinogram_cp, volume_cp, grad_view_sino, *args_fp)
        grad_views = cp.sum(grad_view_sino[...,1:-1,:], axis=-2, keepdims=False)  # over u, clip borders
        grad_views = cp.sum(grad_views[1:-1], axis=0, keepdims=False)  # over v -> shape (Np, 16)
        grad_views = cp.reshape(grad_views, views_cp.shape)
        grad_views = dlpack.from_dlpack(grad_views)
        print(grad_views)

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, cuda_kernels: (ProjectorKernel, BackprojectorKernel),
                sinogram:torch.Tensor, centers_time, volume: torch.Tensor, voxel_scale,
                sampled_projections, trajectory, views, sampling_rate, relaxation):
        """Run the CUDA ray-trace kernel to project ``volume`` into ``sinogram`` for the given
        ``views``/geometry, converting tensors to/from CuPy via DLPack. Saves ``volume``, ``views``,
        ``sampled_projections`` and the other forward-pass config on ``ctx`` for ``backward``."""
        projector = cuda_kernels[0]  # type: ProjectorKernel
        device = sinogram.device
        sinogram_cp = cp.from_dlpack(dlpack.to_dlpack(sinogram.ravel().to('cuda')))  # Todo: check if contiguous sinogram is needed
        volume = volume.squeeze(0)  # No num_particles dimension
        centers_time = cp.asarray(centers_time.detach())

        ctx.save_for_backward(volume, views, sampled_projections)
        ctx.centers_time = centers_time
        ctx.voxel_scale = voxel_scale
        ctx.trajectory = trajectory
        ctx.sampling_rate = sampling_rate
        ctx.relaxation = relaxation
        ctx.cuda_kernels = cuda_kernels

        # Run ray tracing
        volume_cp =  cp.from_dlpack(dlpack.to_dlpack(volume.detach().to('cuda')))
        args_fp = RayTraceProjector.args_fp(ctx, volume_cp, sinogram, centers_time,
                                            sampled_projections, voxel_scale, trajectory, views, sampling_rate)
        vol_tex = projector.create_texture(volume_cp)
        projector(sinogram.shape, vol_tex.ptr, sinogram_cp, *args_fp)  # updates sinogram

        # Convert to torch.Tensor
        sinogram = dlpack.from_dlpack(sinogram_cp).reshape(sinogram.shape).to(device)
        return sinogram

    @staticmethod
    def backward(ctx, *grad_output):
        """Run the CUDA backprojection kernel to compute the volume gradient from the sinogram
        gradient, then (if the views require grad) the numerical geometry gradient via
        ``numeric_geom_gradient``. Returns a gradient tuple aligned with ``forward``'s inputs
        (``None`` for the non-differentiable ones: cuda_kernels, sinogram, centers_time,
        voxel_scale, sampled_projections, trajectory, sampling_rate, relaxation)."""
        projector, backprojector = ctx.cuda_kernels  # type: (ProjectorKernel, BackprojectorKernel)
        relaxation = ctx.relaxation
        sino_shape = grad_output[0].shape
        volume, views, projection_indices = ctx.saved_tensors
        grad_sino_cp = cp.from_dlpack(dlpack.to_dlpack(relaxation * grad_output[0].to('cuda')))  # dtype: cp.float32
        volume_cp = cp.from_dlpack(dlpack.to_dlpack(volume.detach().to('cuda')))

        # Backprojection
        grad_volume_cp = cp.zeros(volume.shape, dtype = cp.float32)
        args_bp = RayTraceProjector.args_bp(ctx, volume, grad_output[0], ctx.centers_time, projection_indices, views)
        residual_tex = backprojector.create_texture(grad_sino_cp)
        backprojector(grad_volume_cp.shape, residual_tex.ptr, grad_volume_cp, *args_bp)
        # Convert to torch.Tensor
        grad_volume = dlpack.from_dlpack(grad_volume_cp).reshape(grad_volume_cp.shape).to(volume.device)
        grad_volume = grad_volume / relaxation  # learning rate will be refactored again in optimizer.step
        grad_volume = grad_volume.unsqueeze(0)  # reintroduce num_particles dimension

        # Numerical geometry gradient
        args_fp = RayTraceProjector.args_fp(ctx, volume, grad_sino_cp, ctx.centers_time, projection_indices,
                                            ctx.voxel_scale, ctx.trajectory, views, ctx.sampling_rate)
        volume_tex = projector.create_texture(volume_cp)
        grad_views = RayTraceProjector.numeric_geom_gradient(projector, grad_sino_cp, ctx.views_requires_grad,
                                                             grad_output[0].device, (volume_tex.ptr,),
                                                             sino_shape, args_fp, eps = 1e-3, bounds = 1)

        return None, None, None, grad_volume, None, None, None, grad_views, None, None


class RayTraceCTSimulation(ArrayCTSimulation):
    """CTModule wrapper around ``RayTraceProjector`` (or a subclass, via ``torch_function``): runs
    the CUDA ray-trace kernel each forward pass to add this component's static-volume contribution
    into the sinogram. This is the class actually exposed as ``ctrex.projectors.RayTraceCTSimulation``
    (see the note in ``ctrex/projectors/__init__.py`` about the array_projectors version of the
    same name being shadowed)."""
    torch_function = RayTraceProjector

    def __init__(self, sample_component, sampling_rate = 1):
        """Set up the forward/backprojection CUDA kernels used by ``torch_function``."""
        nn.Module.__init__(self)
        ArrayCTSimulation.__init__(self, sample_component)
        self.sampling_rate = sampling_rate
        self.projector = ProjectorKernel()
        self.backprojector = BackprojectorKernel()
        self.cuda_kernels = (self.projector, self.backprojector)

    @property
    def relaxation(self):
        """The shape model's 'attenuations' learning rate, reused here as the SIRT relaxation
        factor (defaults to 1 if not set)."""
        return self.shape_model.learning_rates.get('attenuations', 1)

    def forward(self, sinogram, views, sampled_projections):
        """Ray-trace ``volume`` (from the shape model) into a temporary sinogram via
        ``torch_function.apply`` and add it into ``sinogram``; skipped entirely (with a printed
        notice) when ``torch.jit.is_tracing()``, since the CUDA kernels aren't traceable."""
        # temporary tensors are created here if necessary with correct device
        track_time = self.ct_trajectory.projection_time(sampled_projections)
        centers_time = self.track_model(track_time)

        voxel_scale = self.shape_model.voxel_scale
        sinogram_add = 0 * sinogram  # type: torch.Tensor

        if torch.jit.is_tracing():
            print("Not doing ray tracing in a traced model")
        else:
            sinogram_add = self.torch_function.apply(self.cuda_kernels,
                                                     sinogram_add, centers_time, *self.shape_model.shape_params, voxel_scale,
                                                     sampled_projections, self.ct_trajectory, views, self.sampling_rate,
                                                     self.relaxation)
        sinogram = sinogram + sinogram_add
        return sinogram


class SIRTProjector(RayTraceProjector):
    """Same forward pass as ``RayTraceProjector``, but with a ``backward`` that applies SIRT-style
    ray and voxel preconditioning around the backprojection, so the resulting volume gradient
    corresponds to a (preconditioned) least-squares/SIRT update rather than a plain adjoint."""

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor):
        """Precondition the sinogram residual with ``ray_weighting`` (``R(Ax-y)``), backproject it
        (``A^T R(Ax-y)``), then precondition the resulting volume gradient with
        ``voxel_weighting`` if present (``C A^T R(Ax-y)``); also computes the numerical geometry
        gradient as in ``RayTraceProjector.backward``."""
        projector, backprojector, ray_weighting, voxel_weighting = ctx.cuda_kernels  # type: (CudaKernel,)*4
        sinogram = grad_output[0]
        relaxation = float(ctx.relaxation)
        volume, views, projection_indices = ctx.saved_tensors
        grad_sino_cp = cp.from_dlpack(dlpack.to_dlpack(relaxation * grad_output[0].to('cuda')))  # dtype: cp.float32

        # Apply the SIRT-style preconditioning if loss is L2 loss:
        args_fp = RayTraceProjector.args_fp(ctx, ctx.trajectory.volume, grad_sino_cp, ctx.centers_time, projection_indices,
                                            ctx.voxel_scale, ctx.trajectory, views, ctx.sampling_rate)
        ray_weighting(sinogram.shape, grad_sino_cp, *args_fp)  # R (Ax - y)

        grad_volume_cp = cp.zeros(volume.shape, dtype = cp.float32)
        args_bp = RayTraceProjector.args_bp(ctx, grad_volume_cp, sinogram, ctx.centers_time, projection_indices, views)
        residual_tex = backprojector.create_texture(grad_sino_cp)
        backprojector(grad_volume_cp.shape, residual_tex.ptr, grad_volume_cp, *args_bp)  # A^T R (Ax - y)

        # Voxel weighting
        if voxel_weighting is not None:
            voxel_weighting(grad_volume_cp.shape, grad_volume_cp, *args_bp)  # C A^T R (Ax - y)

        # Convert to torch.Tensor
        grad_volume = dlpack.from_dlpack(grad_volume_cp).reshape(grad_volume_cp.shape).to(volume.device)
        grad_volume = grad_volume / relaxation
        grad_volume = grad_volume.unsqueeze(0)  # reintroduce num_particles dimension

        # Numerical geometry gradient
        volume_cp = cp.asarray(volume.detach())
        volume_tex = projector.create_texture(volume_cp)
        args_fp = RayTraceProjector.args_fp(ctx, volume, grad_sino_cp, ctx.centers_time, projection_indices,
                                            ctx.voxel_scale, ctx.trajectory, views, ctx.sampling_rate)
        grad_views = RayTraceProjector.numeric_geom_gradient(projector, grad_sino_cp, ctx.views_requires_grad,
                                                             grad_output[0].device, (volume_tex.ptr,),
                                                             sinogram.shape, args_fp, eps = 1e-3, bounds = 1)

        return None, None, None, grad_volume, None, None, None, grad_views, None, None


class SIRTCTSimulation(RayTraceCTSimulation):
    """``RayTraceCTSimulation`` variant that reconstructs with SIRT-style preconditioning:
    swaps in ``SIRTProjector`` and adds the ray/voxel weighting kernels it needs."""
    torch_function = SIRTProjector

    def __init__(self, sample_component, sampling_rate=1):
        """Add the ray-weighting and voxel-weighting CUDA kernels used by ``SIRTProjector``."""
        super().__init__(sample_component, sampling_rate)
        self.ray_weighting = RayweightingKernel()
        self.voxel_weighting = VoxelweightingKernel()
        self.cuda_kernels = self.projector, self.backprojector, self.ray_weighting, self.voxel_weighting


def check_geom_grad():
    """Standalone dev/debug demo comparing the analytic geometry-gradient kernel
    (``GeomBackpropKernel``) against a numerical finite-difference gradient, using hard-coded
    example geometry/detector/volume values. Prints both gradients for manual comparison.

    NOTE: not currently used/called anywhere outside this file's own ``if __name__ == '__main__'``
    block below - looks like a standalone dev/debug utility rather than something exercised by the
    rest of the codebase.
    """
    projector = ProjectorKernel()
    geombackpropagator = GeomBackpropKernel()
    detector_shape = (1896, 1520)
    detector_rshape = (14, 1520)
    num_projections = 2
    sino_shape = (detector_rshape[0],num_projections,detector_rshape[1])
    views = torch.tensor([[[[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                            [0.0000e+00, 1.0000e+00, 0.0000e+00, 4.5000e+02],
                            [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
                            [3.0500e-02, -1.8300e+01, 0.0000e+00, 1.0000e+00]]],
                          [[[-9.1227e-01, -4.0959e-01, 0.0000e+00, -1.8432e+02],
                            [4.0959e-01, -9.1227e-01, 0.0000e+00, -4.1052e+02],
                            [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
                            [7.4677e+00, 1.6707e+01, 0.0000e+00, 1.0000e+00]]],
                          ])
    views_cp = cp.asarray(views.contiguous().detach(), dtype=cp.float32)
    centers_time_cp = cp.array([[[760, 760, 948]]*2], dtype = cp.float32)
    pixel_size = 0.15
    min_distance = 2557.368408203125
    step_size = 2.0
    max_distance = 3632.170654296875
    voxel_scale = 2
    voxel_size = 0.005913202650845051
    roi_u_start = 0
    roi_v_start = 941
    args_fp = (detector_shape, views_cp, centers_time_cp, pixel_size,
               min_distance, step_size, max_distance, voxel_scale, voxel_size, roi_u_start, roi_v_start)
    grad_output = (1+0*torch.normal(0, 1, (detector_rshape[0],num_projections,detector_shape[1]), device = torch.device('cuda')),)
    volume_cp = cp.asarray(torch.normal(0, 1, (detector_rshape[0]//2,detector_rshape[0]//2,detector_shape[1]//2)))
    volume_tex = projector.create_texture(volume_cp)

    grad_sinogram_cp = cp.asarray(grad_output[0].to(torch.float32).detach().contiguous(), dtype=cp.float32)
    grad_view_sino = cp.tile(grad_sinogram_cp[..., cp.newaxis] * 0, (1, 1, 1, 16))  # (Nv, Np, Nu, 16)
    geombackpropagator(sino_shape, volume_tex, grad_sinogram_cp, grad_view_sino, *args_fp)
    grad_views = cp.sum(grad_view_sino[..., 1:-1, :], axis=-2, keepdims=False)  # over u, clip borders
    grad_views = cp.sum(grad_views[1:-1], axis=0, keepdims=False)  # over v -> shape (Np, 16)
    grad_views = cp.reshape(grad_views, views_cp.shape)
    grad_views = dlpack.from_dlpack(grad_views)
    print(grad_views)

    eps = 1e-3
    grad_num = torch.zeros_like(grad_views)
    for view_i in range(4):
        for view_j in range(4):
            views_delta_cp = cp.asarray(views.contiguous().detach(), dtype=cp.float32)
            views_delta_cp[...,view_i,view_j] += eps
            sinogram = cp.zeros(sino_shape, dtype = cp.float32)
            sinogram_delta = cp.zeros(sino_shape, dtype = cp.float32)
            projector(sino_shape, volume_tex, sinogram_delta, views_delta_cp, *args_fp[1:])
            projector(sino_shape, volume_tex, sinogram, views_cp, *args_fp[1:])
            grad_sino_num = grad_output[0] * (sinogram_delta - sinogram) / eps
            grad_sino_num = cp.sum(grad_sino_num[..., 1:-1], axis=-1, keepdims=False)  # over u, clip borders
            grad_sino_num = cp.sum(grad_sino_num[1:-1], axis=0, keepdims=False)  # over v -> shape (Np, 1)
            grad_num[:,0,view_i,view_j] = torch.tensor(grad_sino_num)  # 0 subviews
    print(grad_num)
    pass


if __name__ == '__main__':
    check_geom_grad()
