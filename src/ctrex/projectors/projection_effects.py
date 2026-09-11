"""Sinogram-domain projection effects: CTModule projectors that model detector/beam artifacts
(heel effect, beam intensity fluctuations, background noise, detector masking) instead of
projecting a sample component. Applied like any other CTModule in a SequentialCTModule pipeline,
usually without a registered sample_component (they act only on the sinogram passed in).
"""
import torch
from torch import nn

from ctrex.projectors.base_modules import CTModule
from ctrex.optimization import torchtools as tt


class CTEffects(CTModule):
    """Base/no-op class for sinogram-domain effects; passes the sinogram through unchanged.
    Used directly as a placeholder projector (e.g. 'nothing' entries in a pipeline) and
    subclassed by the actual effects below."""
    def __init__(self, learning_rate = None):
        super().__init__(None, learning_rate = learning_rate)

    def forward(self, sinogram, views, sampled_projections):
        """Return ``sinogram`` unchanged."""
        return sinogram


class HeelEffect(CTEffects):
    """Models the anode heel effect: X-ray intensity varies across the detector width due to
    beam-path-length differences through the anode. Applies a learnable polynomial correction
    (in normalized detector-u coordinate around the centre of rotation) whose magnitude also
    depends on the local optical depth."""
    def __init__(self, power=(1.0, 0.0, 0.0), magnitude=0.05, learning_rate = None):
        super().__init__(None)

        # Learnable power and slope
        self.magnitude = nn.Parameter(torch.tensor(magnitude, dtype=torch.float32))  # magnitude
        self.power0 = nn.Parameter(torch.tensor(power[0], dtype=torch.float32))
        self.power1 = nn.Parameter(torch.tensor(power[1], dtype=torch.float32))
        self.power2 = nn.Parameter(torch.tensor(power[2], dtype=torch.float32))  # polynomial magnitude(0 + 1 * y^1 + p y^2)
        # Setting learning rates should be done after nn.Module init
        self.learning_rates = learning_rate

    def forward(self, sinogram, views, sampled_projections):
        """
        sinogram: Tensor [..., detector_width]  (last dimension = detector axis)
        cor: scalar or None. If None, use half width of detector.
        """
        detector_width = sinogram.shape[-1]
        device = sinogram.device
        cor = self.ct_trajectory.centre_of_rotation
        # cor = self.cor if self.cor is not None else detector_width // 2

        # Detector coordinates
        u = torch.arange(detector_width, dtype=torch.float32, device=device)
        u_norm = (u - cor) / (detector_width / 2)  # between -1 and +1 for central cor

        # Apply polynomial correction depending on projected optical depth
        sinogram = torch.relu(sinogram)
        power_corr = self.power0 + self.power1 * torch.relu(sinogram) + self.power2 * sinogram ** 2
        sinogram_heel = sinogram + torch.clamp(power_corr, 0, None) * self.magnitude * u_norm

        return sinogram_heel

    @tt.register_loss
    def heel_power(self):
        """Regularization loss penalizing the polynomial correction going negative over a sampled
        range of optical depths, registered as an optimization loss via ``tt.register_loss``."""
        optical_depth = torch.linspace(0, 2, 100)
        power_corr = 1 + self.power1 * torch.abs(optical_depth) + self.power2 * optical_depth **2
        loss = torch.sum(torch.clamp(power_corr, None, 0)**2)
        return loss


class BeamFluctuations(CTEffects):
    """Models slow drift in X-ray source intensity over the scan: holds a (learnable, currently
    commented-out) per-projection additive offset, interpolated up from sparsely measured
    ``fluctuations`` at ``projection_indices`` to one value per projection, and added into the
    sinogram at the sampled projections."""
    def __init__(self, fluctuations = 0., projection_indices = None, num_projections = None,
                 learning_rate = None, loss_weight = 0):
        super().__init__(None, loss_weight)
        device = fluctuations.device if hasattr(fluctuations, 'device') else torch.get_default_device()

        # self.fluctuations = nn.Parameter(torch.as_tensor(fluctuations, dtype = torch.float32))
        # self.projection_indices = (projection_indices if projection_indices is not None
        #                            else torch.arange(0, len(self.fluctuations), device = self.fluctuations.device))
        if num_projections is None:
            num_projections = projection_indices[-1]

        self.projection_indices = torch.as_tensor(projection_indices, dtype=torch.float32, device=device)
        fluctuations = torch.as_tensor(fluctuations, dtype=torch.float32, device=device)
        # assume regular spacing of measured fluctuations, including end point
        self.fluctuations = nn.functional.interpolate(fluctuations.view(1, 1, -1), int(num_projections)).view(1, -1, 1)
        self.learning_rates = learning_rate

    def forward(self, sinogram, views, sampled_projections):
        """Add the interpolated fluctuation value for each of ``sampled_projections`` into
        ``sinogram``."""
        # self.fluctuations = self.fluctuations - self.fluctuations.mean()
        # device = sampled_projections.device
        fluctuations = self.fluctuations.to(sinogram.device)
        sampled_fluctuations = fluctuations[:,sampled_projections]
        sinogram = sinogram + sampled_fluctuations.to(sinogram.device)
        return sinogram


    @tt.register_loss
    def beam_fluctuations(self):
        """Regularization loss combining a ridge penalty (keep fluctuations small/centered around
        their mean) and a temporal-smoothness penalty (penalize large jumps between consecutive
        projections' fluctuations), registered as an optimization loss via ``tt.register_loss``."""
        # Ridge
        magnitude = ((self.fluctuations - self.fluctuations.mean())**2).sum()
        # Temporal smoothness
        time_variations_l2 = ((self.fluctuations[1:] - self.fluctuations[:-1])**2).sum()
        loss = magnitude + time_variations_l2
        return loss


class CTNoise(CTEffects):
    """Adds background noise (Poisson or Gaussian, or none) to the sinogram, with a mean/std that
    can either be set directly or estimated from the data (``estimate_noise_from_sides``,
    ``estimate_noise_from_projection``)."""
    background_type_none = "None"
    background_type_poisson = "poisson"
    background_type_gaussian = "gaussian"

    def __init__(self, background_type = "None", background_mean = 0, background_std = 0):
        super().__init__(None)
        self.bg_type = background_type
        self.background_mean = background_mean
        self.background_std = background_std

    @torch.no_grad()
    def estimate_noise_from_sides(self, sinogram, width = 10, num_projections = 10):
        """Estimate and store ``background_mean``/``background_std`` from the leftmost/rightmost
        ``width`` detector columns of the first ``num_projections`` projections, assuming those
        border regions contain no sample (air only)."""
        #naive estimate of noise taking width voxels on left and right of scan, assuming no particles there
        #TODO sample all projections?
        left_air = sinogram[:, 0:num_projections][...,0:width]
        right_air =  sinogram[:, 0:num_projections][...,-width:]
        all_air = torch.concatenate([left_air.flatten(), right_air.flatten()])
        self.background_mean = all_air.mean().item()
        self.background_std = all_air.std().item()
        print(f"Estimated background noise - Mean: {self.background_mean}; Std: {self.background_std}")

    @torch.no_grad()
    def estimate_noise_from_projection(self, sinogram, projection = 0):
        """
        Estimates the mean and standard deviation of Non-Zero Mean Additive White
        Gaussian Noise (AWGN) in a 2D image using the Median Absolute Deviation
        (MAD) of the mean-subtracted Haar Wavelet Diagonal Detail (HH) coefficients.
        #TODO update to work with more projections

        NOTE: does not appear to be called anywhere else in this repo (unlike
        ``estimate_noise_from_sides``, which is used by the multiframe recon scripts) - looks
        like an alternative/unused noise-estimation method.
        """
        def estimate_noise_mean_by_mode(projection, bins=100):
            """
            Estimates the noise mean (mu_N) by finding the mode (peak) of the image histogram
            using PyTorch's torch.histc.
            """
            image_data = projection.flatten().float()
            min_val = image_data.min()
            max_val = image_data.max() + 1e-6
            hist_counts = torch.histc(image_data, bins=bins, min=min_val, max=image_data.max() + 1e-6)
            max_count_index = torch.argmax(hist_counts)
            bin_width = (max_val - min_val) / bins
            estimated_mode = min_val + (max_count_index.item() * bin_width) + (bin_width / 2.0)
            return estimated_mode

        projection = sinogram[:,projection].squeeze()
        estimated_mean = estimate_noise_mean_by_mode(projection)

        data_input = projection.unsqueeze(0).unsqueeze(0).float()
        hh_kernel_value = torch.tensor([[-1., -1.],[1., 1.]], device = projection.device) / 2.0
        kernels = hh_kernel_value.reshape(1, 1, 2, 2).float()
        dwt_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, bias=False, device = projection.device)
        dwt_layer.weight = nn.Parameter(kernels)
        dwt_layer.requires_grad_(False)
        dwt_output = dwt_layer(data_input)
        hh_coefficients = dwt_output[0, 0, :, :]
        flat_hh_coeffs = hh_coefficients.flatten()
        hh_mean = torch.median(flat_hh_coeffs)
        hh_demeaned = flat_hh_coeffs - hh_mean
        abs_hh_demeaned = torch.abs(hh_demeaned)
        mad_hh_demeaned = torch.median(abs_hh_demeaned)
        SCALING_FACTOR = 0.6745
        estimated_std = mad_hh_demeaned / SCALING_FACTOR

        self.background_std = estimated_std.item()
        self.background_mean = estimated_mean.item()
        print(f"Estimated background noise - Mean: {self.background_mean}; Std: {self.background_std}")

    def add_noise(self, sinogram):
        """Add background noise of type ``bg_type`` (none/poisson/gaussian), parameterized by
        ``background_mean``/``background_std``, to ``sinogram``, then clamp the result to be
        non-negative."""
        # possibility to optimise bg_mu, bg_std if included in forward projection
        if self.bg_type == CTNoise.background_type_none:
            return sinogram

        # add background noise on top
        bg_noise = 0
        with torch.no_grad():
            if self.bg_type == CTNoise.background_type_poisson:  # poisson background noise
                bg_noise = torch.poisson(torch.full_like(sinogram,self.background_mean))
            elif self.bg_type == CTNoise.background_type_gaussian:  # gaussian background noise
                bg_noise = torch.randn(sinogram.shape, dtype=torch.float32, device = sinogram.device, requires_grad = False)
                bg_noise = bg_noise * self.background_std + self.background_mean
        sinogram = sinogram + bg_noise
        sinogram = torch.clamp(sinogram, min = 0)  # TODO should this be clamped?
        return sinogram

    def forward(self, sinogram, views, sampled_projections):
        """Add background noise into ``sinogram`` (see ``add_noise``)."""
        sinogram = self.add_noise(sinogram)
        return sinogram


class Mask(torch.autograd.Function):
    """Multiplicative masking of the sinogram with a custom backward pass: instead of the usual
    chain-rule gradient (which would multiply the incoming gradient by the mask again), it divides
    by the mask so that masked-out (zero) sinogram regions don't suppress the gradient reaching
    the projectors upstream, and cleans up the resulting NaN/Inf values. Used by ``DetectorMask``."""
    @staticmethod
    def forward(ctx, sinogram, mask):
        """Multiply ``sinogram`` by ``mask`` elementwise and save ``mask`` for backward."""
        sinogram = sinogram * mask
        ctx.mask = mask
        return sinogram

    @staticmethod
    def backward(ctx, *grad_output):
        """Divide the incoming sinogram gradient by ``mask`` (rather than multiplying, as a plain
        elementwise-multiply backward would) and zero out any resulting NaN/Inf values."""
        grad_sino = grad_output[0]  # type: torch.Tensor
        # remove all pixels that want to backproject an infinite gradient. Maybe there was no beam there?
        # grad_sino[torch.isinf(grad_sino)] = 0
        grad_input = grad_sino / ctx.mask
        # replace all nan, inf residuals with zero values
        grad_input.nan_to_num_(0, 0, 0)
        # multiply with ones to broadcast shape
        # mask = torch.ones_like(grad_sino, dtype = torch.bool) * ctx.mask.isnan()  # type: torch.Tensor
        # Nan grad_sino due to masking becomes 0
        # grad_input[mask] = grad_input[mask].nan_to_num(0)
        return grad_input, *grad_output[1:], None  # None for mask


class DetectorMask(CTEffects):
    """Applies a fixed detector mask (e.g. marking dead/faulty pixels) to the sinogram within a
    given region of interest, using ``Mask`` for a gradient-friendly backward pass."""
    def __init__(self, detector_mask, roi = (slice(None), slice(None))):
        super().__init__(None)
        self.detector_mask = torch.as_tensor(detector_mask)
        self.roi = roi

    def forward(self, sinogram, views, sampled_projections):
        """Multiply ``sinogram`` by the (ROI-cropped) detector mask, adding a time dimension to
        the mask if ``sinogram`` has one."""
        mask_roi = self.detector_mask[self.roi].to(sinogram.device)
        if sinogram.ndim == 3:
            mask_roi = mask_roi.unsqueeze(1)  # time dimension
        sinogram = Mask.apply(sinogram, mask_roi)
        return sinogram
