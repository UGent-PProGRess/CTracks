"""Loss and metric functions for comparing simulated and measured sinograms/volumes.

Includes simple data-fidelity losses (MSE variants, NCC, PSNR), a differentiable
SSIM implementation, edge/gradient-based losses (difference-of-Gaussians, total
variation), and helpers for computing per-particle losses on small sinogram patches.
Several of the more elaborate losses here (the combined MSE/SSIM/NCC loss, PSNR,
the masked losses, per-particle patch losses) are not currently wired up as the
active ``data_loss_func`` anywhere in the pipeline - see individual docstrings.
"""
import torch
from torch.nn import functional as F
import numpy as np
import itertools


def MSE_SSIM_NCC(simulated_sinogram, sinogram, alpha = 0.33, beta = 0.33, gamma = 0.34, mse_params = {}, ssim_params = {}):
    """Weighted combination of multiscale MSE, SSIM and NCC losses (weights alpha/beta/gamma),
    with the SSIM and NCC terms each rescaled by the MSE value so their magnitudes are
    comparable to it.

    NOTE: not currently used/called anywhere - no script or module passes this as a
    ``data_loss_func``.
    """
    mseloss = 0
    ssimloss = 0
    nccloss = 0
    combined_loss = 0
    if alpha > 0:
        # mseloss = MSEloss(simulated_sinogram, sinogram, **mse_params)
        mseloss = MSEloss_multiscale(simulated_sinogram, sinogram, **mse_params)
        combined_loss += alpha * mseloss
    if beta > 0:
        ssimloss = ssim(simulated_sinogram, sinogram, **ssim_params)
        if alpha > 0:
            scaling = mseloss  # / (1-ssimloss)  # rescale ssim loss to be comparable to mse loss, ssim is between 0-1 (0-2?) so just multiply by mseloss?
            ssimloss *= scaling
        combined_loss += beta * ssimloss
    if gamma >0:
        nccloss = NCCloss(simulated_sinogram, sinogram)
        if alpha > 0:
            scaling = mseloss
            nccloss *= scaling
        combined_loss += gamma * nccloss
    return combined_loss


def MSEloss(simulated_sinogram, sinogram, reduction = 'sum', **kwargs):
    """Thin wrapper around ``F.mse_loss``. Only reached internally via ``MSEloss_multiscale``,
    which is itself only used by the currently-unused ``MSE_SSIM_NCC``."""
    loss = F.mse_loss(simulated_sinogram, sinogram, reduction = reduction, **kwargs)
    # loss = torch.mean((simulated_sinogram[...,1:-1,1:-1] - sinogram[...,1:-1,1:-1]) ** 2)  # exclude boundaries
    return loss


def mse_nan(simulated_sinogram, sinogram, reduction = 'sum', eps = 1e-10):
    """
    import matplotlib.pyplot as plt
    plt.imshow(simulated_sinogram[:,0].detach().cpu().numpy(), aspect = 'auto')
    plt.colorbar()
    plt.show()
    :param simulated_sinogram:
    :param sinogram:
    :param reduction:
    :param eps:
    :return:
    """
    diff = (simulated_sinogram - sinogram)**2
    diff[torch.isinf(sinogram)] = eps
    if reduction == 'mean':
        loss = torch.nanmean(diff)
    else:
        loss = torch.nansum(diff)
    return loss


def rebin_int(a: np.ndarray | torch.Tensor, binning):
    """!
    Rebin the data "a" according to integer binning.

    Binning is taking multiple neighboring elements together and replacing
    them by their average.

    @param a: The array on which to bin.
    @param binning: A list of numbers representing the binning-amount in each dimension.
    @return The binned array.
    """
    # rebin arbitrary dimension array
    dimensions = a.ndim
    padding = tuple([(0, (binning[dim] - a.shape[dim]) % binning[dim]) for dim in range(dimensions)])
    a = np.pad(a, padding, 'symmetric')
    newshape = tuple([a.shape[dim] // binning[dim] for dim in range(dimensions)])
    higherdimshape = tuple(itertools.chain(*list(zip(newshape, binning))))  # shape = interleaved.
    a = np.reshape(a, higherdimshape)
    return a.mean(tuple(np.arange(1, 2 * dimensions, 2)))


def rebin(images, dims = (-3, -1)):
    """Downsample ``images`` by a factor of 2 along each of ``dims``, by averaging adjacent
    pairs of elements. Requires the size along each such dim to be even."""
    device = images.device
    for dim in dims:
        half_length = images.shape[dim] // 2
        images = (images.index_select(index = torch.arange(0, half_length*2, 2, device = device), dim = dim)
                  + images.index_select(index = torch.arange(1, half_length*2, 2, device = device), dim = dim)) / 2
    return images

def MSEloss_multiscale(simulated_sinogram, sinogram, scales = (1, 4), dims = (-3, -1), **kwargs):
    """MSE loss accumulated over progressively coarser (2x rebinned) versions of the sinograms,
    up to ``scales[0]`` halvings along ``dims``, stopping early if a dim would shrink to size 1.

    NOTE: only used internally by ``MSE_SSIM_NCC``, which is itself not currently used anywhere.
    """
    loss = MSEloss(simulated_sinogram, sinogram, **kwargs)
    dims = list(dims)
    for scale in range(1, scales[0]):
        if 1 in torch.tensor(sinogram.shape)[list(dims)]:
            break
        simulated_sinogram = rebin(simulated_sinogram, dims)
        sinogram = rebin(sinogram, dims)
        if scale >= scales[0]:
            loss = loss + MSEloss(simulated_sinogram, sinogram, **kwargs)
    return loss

def ssim(img1, img2, window_size=11, size_average=True):
    """Differentiable structural-similarity (SSIM) loss (returns ``1 - SSIM``) between two
    2D or 3D single-channel images, using a Gaussian window convolution to estimate local
    means/variances/covariance.

    NOTE: only used internally by ``MSE_SSIM_NCC``, which is itself not currently used anywhere.
    """

    # Add batch and channel dimensions
    img1 = img1.unsqueeze(0).unsqueeze(0)  # (1, 1, height, width (,depth))
    img2 = img2.unsqueeze(0).unsqueeze(0)  # (1, 1, height, width (,depth))

    if len(img1.shape) == 4:  # 2D case
        conv = F.conv2d
        create_window = create_window_2d
    elif len(img1.shape) == 5:  # 3D case
        conv = F.conv3d
        create_window = create_window_3d

    (_, channel, *spatial_dims) = img1.size()
    real_size = min([window_size] + list(spatial_dims))
    window = create_window(real_size, channel).to(img1.device)

    mu1 = conv(img1, window, padding=0, groups=channel)
    mu2 = conv(img2, window, padding=0, groups=channel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv(img1 * img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = conv(img2 * img2, window, padding=0, groups=channel) - mu2_sq
    sigma12 = conv(img1 * img2, window, padding=0, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean([i for i in range(1, len(ssim_map.shape))])

def create_window_2d(window_size, channel):
    """Build a 2D Gaussian convolution window (sigma=1.5) for ``ssim``, replicated per channel.

    NOTE: only used internally by ``ssim``, which is itself only used by the currently-unused
    ``MSE_SSIM_NCC``.
    """
    def gaussian(window_size, sigma):
        gauss = torch.exp(-torch.pow(torch.arange(window_size).float() - window_size // 2, 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def create_window_3d(window_size, channel):
    """Build a 3D Gaussian convolution window (sigma=1.5) for ``ssim``, replicated per channel.

    NOTE: only used internally by ``ssim``, which is itself only used by the currently-unused
    ``MSE_SSIM_NCC``.
    """
    def gaussian(window_size, sigma):
        gauss = torch.exp(-torch.pow(torch.arange(window_size).float() - window_size // 2, 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1).unsqueeze(2)
    _3D_window = _1D_window * _1D_window.permute(1, 0, 2) * _1D_window.permute(2, 1, 0)
    window = _3D_window.float().unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window


def NCCloss(simulated_sinogram, sinogram):
    """Normalized cross-correlation loss (``(1 - NCC) / 2``, in [0, 1]) between two sinograms,
    computed over their flattened spatial dimensions.

    NOTE: only used internally by ``MSE_SSIM_NCC``, which is itself not currently used anywhere.
    """

    # Flatten the spatial dimensions (D, H, W)
    simulated_sinogram_flat = simulated_sinogram.view(simulated_sinogram.shape[:-3] + (-1,))
    sinogram_flat = sinogram.view(sinogram.shape[:-3] + (-1,))

    mean_simulated = torch.mean(simulated_sinogram_flat, dim=-1, keepdim=True)
    mean_sinogram = torch.mean(sinogram_flat, dim=-1, keepdim=True)
    simulated_sinogram_centered = simulated_sinogram_flat - mean_simulated
    sinogram_centered = sinogram_flat - mean_sinogram

    std_simulated = torch.clamp(torch.std(simulated_sinogram_flat, dim=-1, keepdim=True), min=1e-8)
    std_sinogram = torch.clamp(torch.std(sinogram_flat, dim=-1, keepdim=True), min=1e-8)

    simulated_sinogram_normalized = simulated_sinogram_centered / std_simulated
    sinogram_normalized = sinogram_centered / std_sinogram
    ncc = torch.mean(simulated_sinogram_normalized * sinogram_normalized, dim=-1)
    return (1-ncc)/2

def NCCloss_per_particle(simulated_sinogram, sinogram):
    """Per-particle NCC loss (``(1 - NCC) / 2``): like ``NCCloss`` but keeps the leading
    (particle) dimension instead of reducing over it, returning one loss value per particle.

    NOTE: this is the default ``loss_func`` of ``calculate_particle_losses``, which is itself
    not currently called anywhere.
    """
    # Flatten spatial dims: (N, P_v * H * P_u)
    simulated_flat = simulated_sinogram.view(simulated_sinogram.shape[0], -1)
    target_flat = sinogram.view(sinogram.shape[0], -1)

    # Means
    mean_sim = simulated_flat.mean(dim=1, keepdim=True)
    mean_tgt = target_flat.mean(dim=1, keepdim=True)

    # Centered
    sim_centered = simulated_flat - mean_sim
    tgt_centered = target_flat - mean_tgt

    # Standard deviations (add small epsilon to avoid divide-by-zero)
    std_sim = torch.clamp(sim_centered.std(dim=1, keepdim=True), min=1e-8)
    std_tgt = torch.clamp(tgt_centered.std(dim=1, keepdim=True), min=1e-8)

    # Normalize
    sim_norm = sim_centered / std_sim
    tgt_norm = tgt_centered / std_tgt

    # NCC: mean of element-wise product
    ncc = (sim_norm * tgt_norm).mean(dim=1)  # (N,)

    # Convert to loss in [0, 1]
    return (1 - ncc) / 2  # (N,)

def PSNR(simulated_sinogram, sinogram):
    """Peak signal-to-noise ratio between the two sinograms, using the larger of the two maxima
    as the data range. Asserts the MSE is nonzero (identical sinograms would give infinite PSNR).

    NOTE: not currently used/called anywhere.
    """
    data_range = torch.max(torch.stack([simulated_sinogram.max(), sinogram.max()]))
    mse = MSEloss(simulated_sinogram, sinogram)
    assert mse != 0, "Identical sinograms"
    psnr = 20*torch.log10(data_range/torch.sqrt(mse))
    return psnr



def extract_sinogram_patches(sinogram, u_particles, v_particles, u_offsets, v_offsets):
    """Bilinearly sample small (P_v, P_u) patches out of ``sinogram`` around each particle's
    (u, v) center, for every projection angle, via ``grid_sample``.

    NOTE: only used internally by ``calculate_particle_losses``, which is itself not currently
    called anywhere.

    Args:
        sinogram: (D, H, W) sinogram to sample from.
        u_particles, v_particles: (N, H) per-particle, per-angle patch-center coordinates.
        u_offsets, v_offsets: 1D patch-local offsets defining the (P_u, P_v) patch grid.

    Returns:
        patches: (N, P_v, H, P_u) sampled patches.
    """
    #TODO should be moved somewhere else probably
    D, H, W = sinogram.shape
    N = u_particles.shape[0]
    P_u, P_v = u_offsets.shape[0], v_offsets.shape[0]

    device = sinogram.device
    dtype = sinogram.dtype

    # sinogram: (D, H, W) -> (H, 1, D, W)
    sinogram = sinogram.permute(1, 0, 2).unsqueeze(1)  # (H, 1, D, W)

    # Create meshgrid of patch offsets (P_v, P_u, 2)
    # u_offsets, v_offsets = torch.meshgrid(u_patch_1d, v_patch_1d, indexing='xy')
    patch_offsets = torch.stack([u_offsets, v_offsets], dim=-1).to(device=device, dtype=dtype)  # (P_u, P_v, 2)

    patch_offsets = patch_offsets.permute(1, 0, 2).reshape(1, 1, P_v * P_u, 2)  # (1, 1, P_v*P_u, 2)

    # Stack particle positions: (N, H, 2) then expand to (N, H, P_v*P_u, 2)
    centers = torch.stack([u_particles, v_particles], dim=-1).to(dtype=dtype).unsqueeze(2)
    sample_coords = centers + patch_offsets  # (N, H, P_v*P_u, 2)

    # Normalize grid to [-1, 1]
    sample_coords[..., 0] = (sample_coords[..., 0] / (W - 1)) * 2 - 1  # u / width
    sample_coords[..., 1] = (sample_coords[..., 1] / (D - 1)) * 2 - 1  # v / depth

    # grid: (N*H, P_v, P_u, 2)
    grid = sample_coords.view(N * H, P_v, P_u, 2)

    # Expand sinogram to match (N * H, 1, D, W)
    sinogram = sinogram.expand(H, 1, D, W).unsqueeze(0).expand(N, H, 1, D, W)
    sinogram = sinogram.reshape(N * H, 1, D, W)

    # Sample patches: (N*H, 1, P_v, P_u)
    sampled = F.grid_sample(
        sinogram,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    # Reshape directly to (N, P_v, H, P_u)
    patches = sampled.view(N, H, P_v, P_u).permute(0, 2, 1, 3).contiguous()

    return patches  # (N, P_v, H, P_u)


def calculate_particle_losses(ct_sample, sinogram, simulated_sinogram, sampled_projections,
                              loss_func = NCCloss_per_particle):
    """Compute a per-particle loss (default ``NCCloss_per_particle``) by extracting each
    particle's small local sinogram patch from both the ground-truth and simulated sinograms
    and comparing them.

    NOTE: not currently called anywhere.
    """
    projector = ct_sample.projectors.particles
    trajectory = ct_sample.trajectory

    sampled_projections = trajectory.projection_indices(sampled_projections)
    views = trajectory.calc_trajectory(sampled_projections)

    sino_patches, patch_coordinates = projector.generate_patches(sampled_projections, views)  # will be on particle projector after getting params/models
    ground_truth_patches = extract_sinogram_patches(sinogram, *patch_coordinates)
    recon_patches = extract_sinogram_patches(simulated_sinogram, *patch_coordinates)
    particle_metrics = loss_func(recon_patches, ground_truth_patches)
    return particle_metrics


def gaussian_kernel_1d(sigma, truncate=3.0, device=None, dtype=None):
    """Build a normalized 1D Gaussian kernel with the given ``sigma``, truncated to
    ``truncate * sigma`` on either side of the center."""
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur_spatial_separable_nearest(x, sigma, truncate = 3.0):
    """
    Separable Gaussian blur over the V and U (spatial) dims, replicate-padded at the boundaries.
    x: (V, T, U)
    """
    k1d = gaussian_kernel_1d(
        sigma,
        truncate = truncate,
        device=x.device,
        dtype=x.dtype
    )
    r = k1d.numel() // 2

    # (N, C, V, T, U)
    x_ = x.unsqueeze(0).unsqueeze(0)

    # ---- blur along V, using padding for boundary mode clamp  ----
    x_ = F.pad(x_, (0, 0,   # U
                    0, 0,   # T
                    r, r),  # V
               mode="replicate")
    kv = k1d.view(1, 1, -1, 1, 1)
    x_ = F.conv3d(x_, kv)  # Trims the tensor

    # ---- blur along U , using padding for boundary mode clamp ----
    x_ = F.pad(x_, (r, r,   # U
                    0, 0,   # T
                    0, 0),  # V
               mode="replicate")
    ku = k1d.view(1, 1, 1, 1, -1)
    x_ = F.conv3d(x_, ku)  # Trims the tensor

    return x_.squeeze(0).squeeze(0)


def difference_of_gaussians_torch(
        x: torch.Tensor,
        sigma_small: float,
        sigma_large: float,
        truncate: float = 3.0,
):
    """
    PyTorch implementation of DoG.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (V, T, U)
    sigma_small : float
        Smaller Gaussian sigma
    sigma_large : float
        Larger Gaussian sigma
    truncate : float
        Kernel radius = truncate * sigma

    Returns
    -------
    torch.Tensor
        DoG-filtered tensor of shape (V, T, U)
    """

    g_small = gaussian_blur_spatial_separable_nearest(x, sigma=sigma_small, truncate=truncate)
    g_large = gaussian_blur_spatial_separable_nearest(x, sigma= sigma_large, truncate=truncate)

    return g_small - g_large


def dog_loss(simulated_sinogram, sinogram, sigma_small = 2, sigma_large = 6, reduction = 'mean', **kwargs):
    """MSE loss between the difference-of-Gaussians (band-pass filtered, edge-emphasizing)
    versions of the simulated and ground-truth sinograms."""
    simulated_sinogram = difference_of_gaussians_torch(simulated_sinogram, sigma_small, sigma_large)
    sinogram = difference_of_gaussians_torch(sinogram, sigma_small, sigma_large)
    loss = F.mse_loss(simulated_sinogram, sinogram, reduction = reduction, **kwargs)
    return loss


def _pad_for_bc(x, pad, bc: str):
    """
    pad: (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_front, pad_d_back)
    bc: 'neumann' | 'periodic' | 'dirichlet'
    """
    if bc == 'neumann':
        # Reflect/replicate edge values (zero normal derivative at boundary)
        return F.pad(x, pad, mode='replicate')
    elif bc == 'periodic':
        # Circular padding
        return F.pad(x, pad, mode='circular')
    elif bc == 'dirichlet':
        # Zero padding
        return F.pad(x, pad, mode='constant', value=0.0)
    else:
        raise ValueError(f"Unknown bc: {bc}")

def forward_diffs_3d(volume: torch.Tensor, voxel_scale, bc='neumann'):
    """
    Compute forward differences with same shape as u, using the chosen boundary condition.
    u: [B, C, D, H, W] or [D, H, W]
    Returns dx, dy, dz of the same shape as u.
    """
    added_batch = False
    added_channel = False
    if volume.dim() == 3:
        # [D,H,W] -> [1,1,D,H,W]
        volume = volume.unsqueeze(0).unsqueeze(0)
        added_batch, added_channel = True, True
    elif volume.dim() == 4:
        # [C,D,H,W] -> [1,C,D,H,W]
        volume = volume.unsqueeze(0)
        added_batch = True
    elif volume.dim() != 5:
        raise ValueError("u must be [D,H,W], [C,D,H,W], or [B,C,D,H,W]")

    B, C, D, H, W = volume.shape

    # x-difference (along W) : u[..., :, :, 1:] - u[..., :, :, :-1]
    dx_core = volume[..., :, :, 1:] - volume[..., :, :, :-1]
    # pad one slice on the right to match W
    dx = torch.cat([dx_core, torch.zeros_like(volume[..., :, :, :1])], dim=-1)
    if bc != 'dirichlet':
        # Replace the last padded slice using BC
        # Construct "u at W" via padding and subtract u[..., :, :, -1:]
        # For Neumann, last diff should be 0; for Periodic, wrap-around difference
        if bc == 'neumann':
            dx[..., -1] = 0.0
        elif bc == 'periodic':
            dx[..., -1] = (volume[..., :, :, 0] - volume[..., :, :, -1])
    # For dirichlet, zeros at boundary are fine

    # y-difference (along H)
    dy_core = volume[..., :, 1:, :] - volume[..., :, :-1, :]
    dy = torch.cat([dy_core, torch.zeros_like(volume[..., :, :1, :])], dim=-2)
    if bc != 'dirichlet':
        if bc == 'neumann':
            dy[..., -1, :] = 0.0
        elif bc == 'periodic':
            dy[..., -1, :] = (volume[..., 0, :] - volume[..., -1, :])

    # z-difference (along D)
    dz_core = volume[..., 1:, :, :] - volume[..., :-1, :, :]
    dz = torch.cat([dz_core, torch.zeros_like(volume[..., :1, :, :])], dim=-3)
    if bc != 'dirichlet':
        if bc == 'neumann':
            dz[..., -1, :, :] = 0.0
        elif bc == 'periodic':
            dz[..., -1, :, :] = (volume[..., 0, :, :] - volume[..., -1, :, :])

    if added_batch and added_channel:
        dx, dy, dz = dx[0,0], dy[0,0], dz[0,0]
    elif added_batch and not added_channel:
        dx, dy, dz = dx[0], dy[0], dz[0]

    # Scale by voxel spacing to get physical gradient magnitudes
    dx = dx / voxel_scale
    dy = dy / voxel_scale
    dz = dz / voxel_scale

    return dx, dy, dz

def tv_isotropic(volume, voxel_scale, eps=1e-6, bc='neumann'):
    """
    Isotropic TV: sum sqrt(dx^2 + dy^2 + dz^2 + eps^2)
    voxel_size: (dz, dy, dx)
    """
    dx, dy, dz = forward_diffs_3d(volume, voxel_scale, bc=bc)
    tv_field = torch.sqrt(dx*dx + dy*dy + dz*dz + eps*eps)
    return tv_field


def tv_anisotropic(volume, voxel_scale, bc='neumann', reduction='sum'):
    """
    Anisotropic TV: sum (|dx| + |dy| + |dz|)
    voxel_size: (dz, dy, dx)

    NOTE: not currently called anywhere.
    """
    dx, dy, dz = forward_diffs_3d(volume, voxel_scale, bc=bc)
    tv_field = dx.abs() + dy.abs() + dz.abs()
    return tv_field

def mask_loss(simulated_sinogram, sinogram, reduction = 'sum', u_slice = slice(70, 590)):
    """MSE loss restricted to a fixed slice of the sinogram's ``u`` (last) dimension.

    NOTE: not currently used/called anywhere.
    """
    loss = F.mse_loss(simulated_sinogram[:,:,u_slice], sinogram[:,:,u_slice], reduction=reduction)
    return loss

class Mask_Loss(torch.nn.MSELoss):
    """MSE loss computed only inside (or, if ``invert``, outside) a rectangular (v, u) region
    of interest of the sinogram, masking out the rest before comparing.

    NOTE: not currently instantiated anywhere - only referenced in a commented-out line in
    scripts/recon_multiphase.py.
    """
    __constants__ = ["reduction", "roi", "invert"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean", roi_vu = (slice(None), slice(None)),
                 invert = False) -> None:
        super().__init__(size_average, reduce, reduction)
        self.roi = roi_vu
        self.invert = invert

    def forward(self, simulated_sinogram: torch.Tensor, sinogram: torch.Tensor) -> torch.Tensor:
        """Zero out the sinogram outside (or, if ``self.invert``, inside) ``self.roi`` on both
        inputs, then compute the standard MSE loss on the masked tensors."""
        mask = torch.ones_like(simulated_sinogram[:,:1], dtype = torch.bool)
        mask[self.roi[0], :, self.roi[1]] = 0
        if self.invert:
            mask = ~mask

        return F.mse_loss(simulated_sinogram * mask, sinogram * mask, reduction=self.reduction)
