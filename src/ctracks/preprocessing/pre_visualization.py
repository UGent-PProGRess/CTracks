import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import difference_of_gaussians

from ctrex.utils import datasets as dl
from ctrex.optimization.lossFunctions import difference_of_gaussians_torch


def visualize_max_and_time(sinogram_particles: dl.CTDatasetDifference, t_start = 0, t_end = 1, t_step = 1,
                           proj_per_rot = 850, thresh_intensity = 0.4, cmin = 0.6, high_sigma = 5, intensity_max = None, dog = False,
                           god = True, fig_fmt = "./proj_diff_dog_%s.png"):
    """Visualize where/when particles appear in a difference sinogram: for the projection range
    [t_start, t_end) (in rotations, stepped by t_step), blob-enhance each (v, u) pixel's intensity
    over time (via a difference-of-Gaussians filter, ``dog``=skimage or ``god``=torch variant),
    then color-code each pixel by the time of its peak intensity (colormap) and modulate its
    brightness by that peak's magnitude. Saves and shows the resulting RGB image.

    Returns:
        t_particle: normalized time-of-max-intensity per (v, u) pixel.
        max_intensity: the (thresholded/rescaled) peak intensity per (v, u) pixel.
    """
    sampled_projections = slice(int(t_start * proj_per_rot), int(t_end * proj_per_rot), t_step)
    sinogram_particles_sample = sinogram_particles[sampled_projections]
    norm_intensity = sinogram_particles_sample

    # Normalize intensity and enhance particles by spatial blob detection
    intensity_max = intensity_max or sinogram_particles_sample.max()
    norm_intensity = norm_intensity / intensity_max  # between 0 and 1
    if dog:
        norm_intensity = norm_intensity.cpu().numpy()
        norm_intensity = difference_of_gaussians(norm_intensity, 2, high_sigma, channel_axis=1)
    if god:  # torch reimplementation of the skimage dog filter above, kept to cross-check equivalence
        norm_intensity = difference_of_gaussians_torch(norm_intensity, 2, high_sigma)
        norm_intensity = norm_intensity.cpu().numpy()

    # Time of max norm_intensity indicates when a particle was there
    t_particle = np.argmax(norm_intensity, axis = 1)
    t_particle = t_particle.astype(np.float32) / t_particle.max()
    cmap = mpl.colormaps["turbo"]  # or "turbo", "plasma", etc.
    t_particle_rgb = cmap(t_particle)[..., :3]  # drop alpha channel

    # Modulate color by max norm_intensity per (v, u)
    norm_intensity *= norm_intensity > thresh_intensity
    slope = (1 - cmin) / (1 - thresh_intensity)
    norm_intensity = 1 + (norm_intensity - 1) * slope  # between cmin and 1
    max_intensity = np.max(norm_intensity, axis = 1)
    max_intensity *= max_intensity > cmin
    t_particle_rgb *= max_intensity[..., None]

    fig, ax = plt.subplots(1, 1, figsize = (8, 10))
    plt.title(f't start: {t_start}, t end: {t_end}, t_step: {t_step}')
    ax.imshow(t_particle_rgb)
    label = f"{t_start}_{t_end}_{t_step}"
    plt.savefig(fig_fmt % label)
    plt.show()

    return t_particle, max_intensity
