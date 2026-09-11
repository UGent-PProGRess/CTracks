"""Thin wrapper around CuPy's raw-kernel loading and CUDA texture creation, used as the base class
for the projector/backprojector/ray- and voxel-weighting CUDA kernels in
``ctrex.projectors.static_projectors``.
"""
import os
import cupy as cp
from cupy.cuda import texture, runtime
import torch
from torch import nn
from torch.utils import dlpack

utils_gpu_path = os.path.dirname(__file__)
sdk_path = os.path.join(utils_gpu_path, 'CUDA Samples')

class CudaKernel:
    """Compiles a named CUDA kernel function from a ``.cu`` source file (via ``cp.RawModule``) and
    calls it with an auto-computed block/grid size based on the output shape. Subclasses set
    ``texture_modes`` (addressing/filtering/read mode for ``create_texture``) and may extend
    ``module_options`` (e.g. extra include paths)."""
    @staticmethod
    def kernel(*_):
        """Placeholder invoked before ``__init__`` sets the real compiled kernel function; always
        returns None."""
        return None

    texture_modes = ((None,), None, None)  # should be defined for child classes
    module_options = ('--std=c++14',
                     f'-I {sdk_path}')

    def __init__(self, name, kernel_path):
        """Compile the kernel named ``name`` from the CUDA source at ``kernel_path``."""
        self.module = cp.RawModule(
            code=open(kernel_path, "r").read(),
            options=self.module_options,
            name_expressions=[name]
        )
        self.kernel = self.module.get_function(name)

    @staticmethod
    def blocks_threads(output_shape):
        """Compute the CUDA (threads_per_block, blocks_per_grid) launch configuration needed to
        cover a 3D ``output_shape``, using a fixed 16x4x4 block size."""
        threads_per_block = (16, 4, 4)
        blocks_per_grid = tuple([(output_shape[di] + threads_per_block[di] - 1) // threads_per_block[di]
                                 for di in range(3)])
        return threads_per_block, blocks_per_grid

    def __call__(self, output_shape, *args):
        """Launch the compiled kernel over a grid sized for ``output_shape``, passing ``args``
        through to it."""
        threads_per_block, blocks_per_grid = self.blocks_threads(output_shape)
        return self.kernel(blocks_per_grid, threads_per_block, args)

    def create_texture(self, volume):
        """
        volume: CuPy array (D, H, W), float32
        Returns: texture.TextureObject
        """
        assert volume.ndim == 3 and volume.dtype == cp.float32
        address_modes, filter_mode, read_mode = self.texture_modes

        # Create channel descriptor. First arguments are number of bits for each channel
        chan_desc = texture.ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)  # 2 for float

        # Create CUDAarray from CuPy volume
        cuda_arr = texture.CUDAarray(chan_desc, *volume.shape[::-1])
        cuda_arr.copy_from(volume)

        res_desc = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuda_arr)

        tex_desc = texture.TextureDescriptor(address_modes, filter_mode, read_mode,
                                             None, None, False)

        return texture.TextureObject(res_desc, tex_desc)

    def create_2d_layered_texture(self, sinogram):
        """
        volume: CuPy array (num_projections_subset, num_rows, num_cols), float32
        Returns: texture.TextureObject for cudaTextureType2DLayered
        """
        assert sinogram.ndim == 3 and sinogram.dtype == cp.float32

        num_projections_subset, num_rows, num_cols = sinogram.shape
        address_modes, filter_mode, read_mode = self.texture_modes

        # Create channel descriptor (32-bit float, 1 channel)
        chan_desc = texture.ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)

        # Create CUDAarray with 2D layered format: num_cols, num_rows, num_projections_subset
        cuda_arr = texture.CUDAarray(chan_desc, num_cols, num_rows, num_projections_subset)

        # Copy from CuPy array into CUDAarray
        cuda_arr.copy_from(sinogram)

        # Resource descriptor: tells CUDA this is a layered 2D texture
        res_desc = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuda_arr)

        # Texture descriptor for 2D texture (only 2 address modes needed)
        tex_desc = texture.TextureDescriptor(address_modes, filter_mode, read_mode,
                                             None, None, False)

        # Create texture object (cudaTextureType2DLayered inferred from CUDAarray)
        return texture.TextureObject(res_desc, tex_desc)


class VolumeFilter(torch.autograd.Function):
    """ Example signature for a typical filter function """
    # NOTE: not currently used/called anywhere in this repo - as the docstring above says, this
    # looks like a template/example torch.autograd.Function showing the expected shape of a
    # forward/backward CUDA-kernel pair (matching the (forward_kernel, backward_kernel) convention
    # used by the real projector kernels), rather than an actually-used filter.
    @staticmethod
    def forward(ctx, cuda_kernels: CudaKernel, input_tensor, output_tensor, *args, **kwargs):
        """Run ``forward_kernel`` into ``output_tensor`` and stash the backward kernel/inputs
        needed to compute the gradient."""
        forward_kernel, backward_kernel = cuda_kernels
        forward_kernel(output_tensor.shape, input_tensor, output_tensor, *args)
        ctx.backward_kernel = backward_kernel
        ctx.save_for_backward(input_tensor)
        ctx.args = args

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Run the saved ``backward_kernel`` to compute the gradient w.r.t. the input tensor."""
        input_tensor = ctx.saved_tensors[0]
        grad_input = torch.zeros_like(input_tensor)
        ctx.backward_kernel(ctx.input, *grad_outputs, grad_input, *ctx.args)
        return None, grad_input, None, (None,) * len(ctx.args)

