"""
Convenience methods to get CT simulators with some default settings

"""
from ctrex.utils.templates import *
from ctracks.particle_projectors import ParticleCTSimulation
from ctracks.visualisation.visualize_results import VisualizeResultsTracks

def get_static_matrix_with_particles(ct_simulation: cs.CTSimulation, voxel_scale = 1, roi_shape = None, pore_mask = None,
                                     num_components = (1,1,1)) -> SequentialCTModule:
    """Build a default three-component sample (particles + cylinder array + static volume matrix)
    and wire it into ``ct_simulation`` as a ``SequentialCTModule`` keyed 'particles'/'cylinder'/
    'static_matrix'. Returns the assembled projectors module."""
    vol_shape = ct_simulation.trajectory.volume.vol_shape
    voxel_size = ct_simulation.trajectory.voxel_size
    roi_shape = roi_shape or tuple([size // voxel_scale for size in vol_shape])
    device = ct_simulation.device
    pore_mask = pore_mask or PoreMaskPlain(vol_shape, device)

    # Define components by their track model and shape model
    component_particles = SampleComponent(
        tm.StaticTrack(3, device, vol_shape, learning_rate=0.3e4,
                       bounds=[0] * 6, pore_mask=pore_mask, mask_confines=True),
        sm.SphereShape(3, device, attenuation_range=(0, 0.1), rad_mean=0.032, rad_std=0.005,
                       rad_range=(0.02, 0.1), learning_rate=(1e-1, 1e-1)),  # radius: lr, attenuation: 0
        loss_weight=0)

    component_cylinder = SampleComponent(
        tm.StaticTrack(2, device, vol_shape, learning_rate=0,
                       bounds=[half_size for dim in range(3) for half_size in (vol_shape[2 - dim] / 2,) * 2],
                       mask_confines=False, pore_mask=pore_mask),
        sm.CylinderArray(3, device, (0, 3.0),
                         radii=np.array([2, 2.9]) / 2 / voxel_size,
                         tilt_std=0, bounds=2.0 / 2 / voxel_size, learning_rate=[1e-2, 0.1, 0.1],
                         tilt_range=(-10, 10)),
        loss_weight=0)

    component_volume = SampleComponent(
        tm.StaticTrack(2, device, vol_shape, learning_rate=0,
                       bounds=[half_size for dim in range(3) for half_size in (vol_shape[2 - dim] / 2,) * 2],
                       mask_confines=False, pore_mask=pore_mask, loss_weight=0),
        sm.VolumeArray(3, device, roi_shape, 1, learning_rate=None,
                       attenuation_range=(-0.6, 4.2), loss_weight=0.003),
        loss_weight=1)

    pore_mask.init_pore_mask([component_volume, component_particles, component_cylinder])

    sample_projectors = SequentialCTModule(collections.OrderedDict(
        [('particles', ParticleCTSimulation(component_particles)),
         ('cylinder', ArrayIntersectionCTSimulation(component_cylinder)),
         ('static_matrix', SIRTCTSimulation(component_volume, sampling_rate=1)),
         ]))

    ct_simulation.set_projectors(sample_projectors)  # Sets trajectory of components
    for ci, component in enumerate((component_particles, component_cylinder, component_volume)):
        component.init_component(num_components[ci])

    return sample_projectors


def reconstruct_dataset(ct_dataset: dl.CTDataset,
                        ct_sample: cs.CTSimulation, recon_name, recon_suffix = None,
                        iterations = 1, subset_size = 1, callbacks = (), data_loss_func = F.mse_loss, num_frames = 3,
                        reconstruct = True, learning_rate = 0.5, gt_sample = None, ):
    """Convenience wrapper that sets up a ``VisualizeResultsTracks`` callback, an ordered-subset
    sampler, and a ``CTReconstruction``/Trainer for ``ct_sample``, optionally running it
    immediately. Returns (trainer, reconstructor)."""
    logger = get_logger(ct_dataset)
    visualizer = VisualizeResultsTracks(ct_dataset, recon_name)
    callbacks = (visualizer,) + callbacks

    sampler = samplers.OrderedSubsetSampler(range(ct_dataset.num_projections), subset_size=subset_size)
    reconstructor = CTReconstruction(ct_dataset, ct_sample, sampler, data_loss_func = data_loss_func,
                                     gt_sample=gt_sample, default_lr=learning_rate)
    trainer = get_trainer(reconstructor, iterations, num_frames, logger, callbacks)
    if reconstruct:
        run_trainer(trainer, reconstructor)
    return trainer, reconstructor
