import pandas as pd
import seaborn as sns
import tifffile
import pprint
from ctrex.optimization.visualize_results import *


def params_to_dataframe(ct_sample, track_time, dataset_name, proj_step = 2):
    """Build a long-format dataframe of per-particle track/shape params at each control point,
    interpolated across projections (every ``proj_step``) for plotting, tagged with ``dataset_name``."""
    track_model = ct_sample.projectors.particles.track_model
    shape_model = ct_sample.projectors.particles.shape_model
    num_particles = track_model.num_particles
    tracks = track_model(track_time).detach().cpu().numpy()
    track_time = track_time.detach().cpu().numpy()
    data_t = []
    for ti in range(track_model.num_control_points):
        # interpolate tracks for every proj_step
        for projection_index in range(ti * (len(track_time) // track_model.num_control_points),
                                      (ti + 1) * (len(track_time) // track_model.num_control_points), proj_step):
            data = {}
            for i,label in enumerate(track_model.labels):  # ti corresponds to control points
                data[label] = track_model.control_points[:, ti, i].detach().cpu().numpy()
            for label, param in shape_model.named_parameters():
                data[label] = param.detach().cpu().numpy()
            data['size'] = shape_model.shape_area.detach().cpu().numpy()
            data['particle_index'] = np.arange(0, num_particles)
            data['dataset'] = dataset_name
            data['track_x'] = tracks[:,projection_index,0]
            data['track_y'] = tracks[:,projection_index,1]
            data['track_z'] = tracks[:,projection_index,2]
            data['track_time'] = track_time[projection_index]
            data_t.append(pd.DataFrame(data))
    data = pd.concat(data_t)
    return data


def get_particles_dataframe(ct_sample_truth, ct_sample_recon, ct_sample_init, sampled_projections):
    """Combine the reconstruction's (and, if available, ground-truth's/initialisation's) particle
    dataframes into one, offsetting ``particle_index`` per dataset so they don't collide. Returns
    None if the reconstruction sample has no particle component."""
    if not hasattr(ct_sample_recon.projectors, 'particles'):
        return None
    track_time = ct_sample_recon.trajectory.projection_time(sampled_projections)
    reconstruction = params_to_dataframe(ct_sample_recon, track_time, 'reconstruction')
    
    if ct_sample_truth is not None:
        ground_truth = params_to_dataframe(ct_sample_truth, track_time, 'ground truth')
        # adjust particle_indices
        reconstruction['particle_index'] += ground_truth['particle_index'].max() + 1
    else:
        ground_truth = None

    if ct_sample_init is not None:
        initialisation = params_to_dataframe(ct_sample_init, track_time, 'initialisation')
        initialisation['particle_index'] += reconstruction['particle_index'].max() + 1
    else:
        initialisation = None

    ground_truth_reconstruction = pd.concat((ground_truth, reconstruction, initialisation))
    return ground_truth_reconstruction


def particle_scatter2d(ax, sinogram, ground_truth_reconstruction, clean):
    """Scatter-plot particle centers (and their tracks) in the x-y plane, colored by z and sized
    by particle size, comparing ground truth vs. reconstruction (and their final positions)."""
    sinogram_depth, sinogram_height, sinogram_width = sinogram.shape[-3:]
    ax.set_xlim(0, sinogram_width)
    ax.set_ylim(0, sinogram_width)

    markers = ['s', 'o']

    if ground_truth_reconstruction is not None:
        dataframe_selected = ground_truth_reconstruction[ground_truth_reconstruction['dataset'] != 'initialisation'].copy()
        dataframe_selected.sort_values(['dataset', 'particle_index'], inplace=True)
        sns.scatterplot(dataframe_selected,
                        x='center_x', y='center_y', size='size', hue='center_z', palette='viridis',
                        ax=ax, sizes = (20, 100),
                        markers={'ground truth': markers[0], 'reconstruction': markers[1]},
                        style='dataset')
        sns.scatterplot(dataframe_selected[dataframe_selected['track_time'] == dataframe_selected['track_time'].max()],
                        x='center_x', y='center_y', size='size', hue='center_z', palette='viridis',
                        ax=ax, sizes = (20, 100),
                        markers={'ground truth': markers[0], 'reconstruction': markers[1]},
                        style='dataset')
        sns.lineplot(dataframe_selected,
                     x='track_x', y='track_y', hue='particle_index', ax=ax, sort = False)
    norm = plt.Normalize(0, sinogram_depth)
    scm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    ax.figure.colorbar(scm, ax=ax, orientation='horizontal')

    handles = [
        plt.Line2D([0], [0], marker=markers[0], color='none', markerfacecolor='lightgray', markeredgecolor='lightgray',
                   markersize=10, label='Ground truth'),
        plt.Line2D([0], [0], marker=markers[1], color='none', markerfacecolor='lightgray', markeredgecolor='lightgray',
                   markersize=10, label='Reconstruction')
    ]

    plt.legend(handles=handles, labels=['Ground truth', 'Reconstruction'], prop={'size': 14}, loc='lower left',
               bbox_to_anchor=(0.0, -0.67))

    if clean:
        remove_axes(ax)
    plt.grid()


def particle_scatter3d(ax, sinogram, ground_truth_reconstruction, clean):
    """3D scatter-plot of particle centers, colored by dataset (ground truth / reconstruction /
    initialisation) and sized by particle radius."""
    sinogram_depth, sinogram_height, sinogram_width = sinogram.shape[-3:]
    ax.set_xlim(0, sinogram_width)
    ax.set_ylim(0, sinogram_width)  # reconstructing to square slices
    ax.set_zlim(0, sinogram_depth)

    marker_map = {'ground truth': 'o', 'reconstruction': 's', 'initialisation': '*'}

    # Plot each dataset with different markers
    for dataset, marker in marker_map.items():
        if ground_truth_reconstruction is None:
            continue
        subset = ground_truth_reconstruction[ground_truth_reconstruction['dataset'] == dataset]
        ax.scatter(subset['center_x'], subset['center_y'], subset['center_z'],
                   s=subset['radii'], c=subset['index'], marker=marker)

    # Only plotting marker types now
    handles = [plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='lightgray', markersize=10, label='Ground truth'),
               plt.Line2D([0], [0], marker='s', color='none', markerfacecolor='lightgray', markersize=10, label='Reconstruction'),
               plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='lightgray', markersize=10, label='Initialisation')
               ]

    plt.legend(handles=handles)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if clean:
        remove_axes(ax, remove_ticklines=False)
    plt.grid()


def detach_particle_params(particle_params):
    """Detach and move to CPU a (track_params, shape_params) pair, where shape_params is itself
    a tuple of tensors."""
    particle_params = (particle_params[0].detach().cpu(),
                       tuple([shape_param.detach().cpu() for shape_param in particle_params[1]]))
    return particle_params


def save_results(ground_truth_reconstruction, reconstructed_sinogram = None, process_params = None):
    """Pickle the combined results dataframe (and optionally write the reconstructed sinogram as
    a tif) under ``process_params['outfolder']/process_params['run_prefix']``."""
    os.makedirs(process_params['outfolder'], exist_ok=True)
    ground_truth_reconstruction.to_pickle(str(os.path.join(process_params['outfolder'],
                                                           process_params['run_prefix'], "particle_positions.pkl")))
    if reconstructed_sinogram is not None:
        tifffile.imwrite(str(os.path.join(process_params['outfolder'], process_params['run_prefix'],
                                          "reconstructed_sinogram.tif")), reconstructed_sinogram)
        # np.save(process_params['outfolder'] + process_params['run_prefix'] + "reconstructed_sinogram.npy", reconstructed_sinogram)


def load_results(process_params):
    """Load the combined results dataframe and reconstructed sinogram previously written by
    ``save_results``."""
    ground_truth_reconstruction = pd.read_pickle(str(os.path.join(process_params['outfolder'],
                                                                  process_params['run_prefix'], "particle_positions.pkl")))
    reconstructed_sinogram = tifffile.imread(str(os.path.join(process_params['outfolder'], process_params['run_prefix'],
                                                              "reconstructed_sinogram.tif")))
    return ground_truth_reconstruction, reconstructed_sinogram


def save_params_to_file(dict_list, dict_names, filepath = "./output.txt"):
    """Pretty-print each dict in ``dict_list`` (labeled with the matching entry in ``dict_names``)
    to a text file, for a human-readable record of run configuration."""
    print(f"Saving all parameters to {filepath}")
    dictionaries = [(dict_names[i], dict_list[i]) for i in range(len(dict_list))]
    with open(filepath, 'w') as f:
        for name, d in dictionaries:
            f.write(f"=== {name} ===\n")  # Heading with variable name
            pprint.pprint(d, stream=f)  # Pretty print the dictionary
            f.write("\n\n")  # Add space between dictionaries


class VisualizeResultsTracks(VisualizeResults):
    """VisualizeResults callback specialised for particle-track reconstructions: in addition to
    the base sinogram comparison plots, overlays particle scatter/track plots (2D or 3D) and
    attenuation slices, optionally against a ground-truth and/or initialisation sample."""

    def __init__(self,
                 ct_dataset: CTDataset,
                 recon_name,

                 plot_comparison = [plot_overlap_sinogram, plot_difference_sinogram, plot_ground_truth_sinogram][1],
                 cmap = ['white', 'seismic'][1],
                 max_scaling = None,
                 clean = True,
                 plot_3d = False,
                 truth_sample = None,
                 init_sample = None):
        super().__init__(ct_dataset, recon_name, plot_comparison, cmap, max_scaling, clean)
        self.plot_3d = plot_3d
        self.truth_sample = truth_sample
        self.init_sample = init_sample

    def plot_recon_results(self, ct_sample_recon, sinogram, reconstructed_sinogram, sampled_projections):
        """Build the multi-panel comparison figure: particle scatter/tracks (2D or 3D, optionally
        overlaid on the pore mask), measured vs. reconstructed sinogram/projection comparisons,
        and attenuation slices. Called by the base ``VisualizeResults`` callback machinery."""
        ground_truth_reconstruction = get_particles_dataframe(self.truth_sample, ct_sample_recon, self.init_sample,sampled_projections)

        # Display the original and reconstructed particle positions
        fig = plt.figure(figsize=(12, 6))
        if self.plot_3d:
            ax = fig.add_subplot(1, 3, 2, projection='3d')
            particle_scatter3d(ax, sinogram, ground_truth_reconstruction, self.clean)
        else:
            ax = fig.add_subplot(1, 3, 2)
            try:
                pore_mask = ct_sample_recon.projectors[0].sample_component.track_model.pore_mask
            except AttributeError:
                pore_mask = None
            if isinstance(pore_mask, torch.Tensor):
                ax.imshow(torch.mean(pore_mask.cpu().numpy().float(), dim=0), cmap='gray', origin='lower')
            particle_scatter2d(ax, sinogram, ground_truth_reconstruction, self.clean)

        # plot central difference sinogram
        ax = plt.subplot(2, 3, 1)
        height = sinogram.shape[0]
        angles = ct_sample_recon.trajectory.angles[sampled_projections].detach().cpu().numpy()
        self.plot_comparison(np.max(sinogram[height // 8: - height // 8], axis=0),
                        np.max(reconstructed_sinogram[height // 8: - height // 8], axis=0),
                        ax, self.max_scaling, y_axis=angles, cmap = self.cmap)

        # plot aggregate difference projection
        ax = plt.subplot(2, 3, 4)
        self.plot_comparison(np.max(sinogram, axis=1),
                        np.max(reconstructed_sinogram, axis=1),
                        ax, self.max_scaling, ylabel='v', cmap = self.cmap)

        ax = fig.add_subplot(2, 3, 3)
        attenuations, extent, shape_params, limits = slice_attenuation(ct_sample_recon, 0, [0])  # z
        imshow_colorbar(ax, attenuations, limits, extent = extent)

        ax = fig.add_subplot(2, 3, 6)
        attenuations, extent, shape_params, limits = slice_attenuation(ct_sample_recon, 2, [0])  # x
        imshow_colorbar(ax, attenuations, limits, extent = extent)

        return fig

