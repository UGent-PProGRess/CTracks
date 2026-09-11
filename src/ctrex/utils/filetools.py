"""I/O and preprocessing helpers used across the particle-tracking pipeline: listing and
indexing tiff sequences on disk, reading/writing tiffs with slope/offset metadata so float
data can round-trip through scaled integer storage, flat/dark-field normalisation, building
sino_params from acquisition-settings files, saving simulated projections and reconstructed
particle trajectories for later analysis, and small git/versioning helpers used to tag
output run folders.
"""

from datetime import datetime
from pathlib import Path
import os
import fnmatch

import git
import torch
import numpy as np
import tifffile
from tifffile import TiffFileError

from ctrex.optimization import samplers
from ctrex.sample_description.shape_models import ShapeModel
from ctrex.sample_description.track_models import TrackModel
from ctrex.utils import datasets as dl


def get_format(tiffile):
    """!
    Get the prefix and format of a file without the directory.

    Example: prefix_Z_000186.tif will return prefix_Z_ and '%s%05d.tif'
    @param tiffile: The filename of which you want to know prefix and format. Can be a full path to this file.
    @return prefix and format.
    """
    filename, file_extension = os.path.splitext(os.path.basename(tiffile))
    length = len(filename)
    i = length
    while i > 0 and filename[i - 1].isdigit():
        i -= 1
    prefix = filename[0:i]
    fmt = '%s%0' + str(length-i) + 'd' + file_extension
    return prefix, fmt


def get_index(filename, len_prefix, len_ext):
    """!
    Returns the index hidden in the filename.

    For example: filename 'prefix0009.tif' with len_prefix=6 and len_ext=4 will
    return 9
    @param filename: The filename from which to get the index
    @param len_prefix: How many characters is the prefix in this filename long
    @param len_ext: How many characters is the extension (including point!) in this filename.
    @return index of the given file
    """
    if len_ext == 0:
        return int(filename[len_prefix:])
    return int(filename[len_prefix: - len_ext])


def get_indices(file_list):
    """!
    Given a list of files, return a list of their indices.

    @param file_list: A list of filepaths (strings).
    @return A list of ints, the indices
    """
    if not file_list:
        return []
    filename = file_list[-1].rstrip(os.sep)
    prefix, _ = get_format(filename)
    len_prefix = len(prefix)
    # length of extension. Starting from dot to end, zero if no dot is found
    len_ext = len(filename) - filename.rfind('.') if filename.rfind('.') != -1 else 0
    indices = [get_index(os.path.basename(f.rstrip(os.sep)), len_prefix, len_ext) for f in file_list]

    return indices


def listargsort(indices):
    """!
    argsort for lists: get the invert indices that would sort a list of indices
    e.g. listargsort([1,133,7,13]) -> [0, 2, 3, 1]
    
    @param indices:
    @return
    """
    return sorted(list(range(len(indices))), key=lambda k: indices[k])

def filelist(inputpath, fmt ='prefix_%s.tif', sort = True):
    """!
    Get a list of full paths of all tiffs that fit the path and prefix in inputpath and the format in fmt.

    @param inputpath: The input path and prefix that describes the searched files (prefixPath)
    @param fmt: The format the filename should adhere to after its prefix
    @param sort: Whether to sort the list by file index
    @return list of paths (strings)
    """
    # prefix = fmt.split('_')[0]
    searchpath = os.path.dirname((inputpath + (os.sep if not inputpath.endswith(os.sep) else '')))
    searchpath = searchpath.replace('/',os.sep)
    if not os.path.isdir(searchpath):
        return []
    lookfor = fmt % '*'
    files = fnmatch.filter(os.listdir(searchpath),lookfor)
    if len(files) == 0:
        return []
    if sort:
        try:
            indices = get_indices(files)
            invert_indices = listargsort(indices)  # indices[invert_indices] is sorted
            files = [files[iv] for iv in invert_indices]
        except ValueError:  # didn't find indices to sort
            print("files could not be sorted")
    out = [os.path.join(searchpath,f) for f in files]
    return out


def read_sino_params_from_xre(scan_folder, sino_params = None, device = torch.device('cpu')):
    """Reads sino_params (detector size, source/detector distances, pixel size,
    centre-of-rotation, tilt, projection angles, etc.) from an XRE acquisition-settings text
    file in `scan_folder`, trying the 'Acquisition settings XRE.txt' filename first and
    falling back to 'Dataset settings XRE.txt'.

    NOTE (kept as legacy, currently unused): this consolidates what used to be three separate
    functions - read_acquisition_settings_xre (parsed the raw INI-style settings file into a
    nested dict), update_sino_params_from_file (mapped that dict onto sino_params fields), and
    read_sino_params (tried the two filenames) - along with update_sino_params_from_mask,
    update_sino_params_from_measured_sino and their combiner update_sino_params, which were
    unrelated to reading from XRE files specifically (they filled in sino_params from a mask's
    or measured sinogram's tensor shape instead) and were removed rather than kept, since none
    of the six were called from anywhere in the active pipeline (only each other, or a couple
    of commented-out call sites). This one is kept in case reading sino_params from XRE
    acquisition files is needed again - behaviour is unchanged from the old
    update_sino_params_from_file/read_sino_params pair, just merged into one function.
    """
    for filename in ('Acquisition settings XRE.txt', 'Dataset settings XRE.txt'):
        file_path = os.path.join(scan_folder, filename)
        if os.path.isfile(file_path):
            break
    else:
        raise FileNotFoundError(f"Neither 'Acquisition settings XRE.txt' nor 'Dataset settings XRE.txt' "
                                f"found in {scan_folder}")

    # Parse the INI-style key/value settings file into a nested dict of {section: {key: value}},
    # casting numeric-looking values to float or int and stripping surrounding quotes.
    acq_settings = {}
    current_section = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                acq_settings[current_section] = {}
            elif current_section and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"')  # Remove quotes if present
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass  # keep as string
                acq_settings[current_section][key] = value

    sino_params = sino_params or {}
    ct_params = acq_settings['CT-parameters IN']
    acq_params = acq_settings['Acquisition settings']
    camera_params = acq_settings['Camera settings']

    sino_params['width'] = camera_params['Columns']
    sino_params['height'] = camera_params['Rows']
    sino_params['last angle'] = acq_params['CT stop angle']
    sino_params['sod'] = ct_params['SOD']
    sino_params['sdd'] = ct_params['SDD']
    sino_params['binning'] = camera_params['Binning value']
    sino_params['pixel size'] = ct_params['Pixel size']

    sino_params['horizontal centre'] = ct_params['HC']
    sino_params['vertical centre'] = ct_params['VC']
    sino_params['center_of_rotation'] = ct_params['COR']

    sino_params['tilt'] = ct_params['Tilt']

    sino_params['angles'] = torch.deg2rad(torch.tensor(np.linspace(acq_params['CT start angle'], acq_params['CT stop angle'], acq_params['total projections'], endpoint=True,dtype=np.float32), device=device, dtype=torch.float32))
    return sino_params


def printcounting(printline, currentnr, rangemax):
    """!
    This is for printing something with a 'counter', remaining on a single line.
    @param printline: The string to print.
    @param currentnr: The current counter number.
    @param rangemax: currentnr can be at most rangemax-1.
    @return nothing
    """
    print('\r' + printline, end = '\n' if currentnr == rangemax - 1 else '')

def read_description(filename):
    """Reads a tiff file's dtype and its slope/offset description string (as embedded by
    `write_tiff`/`get_description`). If the file has no readable description (raises
    TiffFileError), falls back to assuming default limits of (0, 1) and synthesizes a
    matching description instead.
    """
    with open(filename, 'rb') as open_file:
        try:
            im = tifffile.imread(open_file)  # type: np.array
            description = tifffile.TiffFile(filename, mode = 'r').pages.first.description
        except TiffFileError:
            limits = (0., 1.)
            im = tifffile.imread(open_file)  # type: np.array
            slope, offset = get_slope_offset(limits, im.dtype)
            description = get_description(slope, offset)
        open_file.close()
        return description, im.dtype

def load_tiff(filename, disable_logger = True):
    """Loads a tiff image and its slope/offset description, temporarily silencing
    tifffile's logger (many of these files trigger noisy warnings) unless disable_logger
    is False.
    """
    logger_disabled = tifffile.logger().disabled
    tifffile.logger().disabled = disable_logger
    description, dtype = read_description(filename)
    with open(filename, 'rb') as open_file:
        im = tifffile.imread(open_file)  # type: np.array
    tifffile.logger().disabled = logger_disabled
    return im, description


def load_3dtiff_slice(filename, slc):
    """Memory-maps a 3D tiff stack and returns only the given slice `slc`, avoiding
    loading the full volume into memory.

    NOTE: not currently used/called anywhere in src/ or scripts/.
    """
    with tifffile.TiffFile(filename) as tif:
        arr = tif.asarray(out = "memmap")
        sub = arr[slc]
    return sub


def get_slope_offset(limits, dtype):
    """Computes the (slope, offset) pair that linearly maps `dtype`'s full integer range
    onto `limits`, used to losslessly store float data as scaled integers in a tiff.
    """
    slope, offset = float(limits[1] - limits[0]) / np.iinfo(dtype).max, float(limits[0])
    return slope, offset


def get_description(slope, offset):
    """Formats a slope/offset pair into the description string embedded in saved tiffs,
    later parsed back by `split_description`.
    """
    description = f'slope = {slope: 6.5E} offset = {offset: 6.5E}'
    return description

def get_limits(description, dtype):
    """Recovers the original (min, max) float limits from a tiff's slope/offset
    description string and its integer dtype - the inverse of `get_slope_offset`.
    """
    slope, offset = split_description(description)
    limits = (offset, offset + slope * np.iinfo(dtype).max)
    return limits

def split_description(description):
    """Parses the slope and offset values out of a description string produced by
    `get_description`.
    """
    slope, offset = float(description.split()[2]), float(description.split()[5])
    return slope, offset


def load_and_scale_tiff(filename, disable_logger = True):
    """Loads a tiff saved via `write_tiff`/`write_tiffs` and rescales it back from its
    stored integer representation to float32 using the slope/offset encoded in its
    description. Returns the rescaled array and the original (min, max) limits.
    """
    im, description = load_tiff(filename, disable_logger)
    slope, offset = split_description(description)
    limits = get_limits(description, im.dtype)
    im = offset + im * slope
    im = im.astype(np.float32)
    return im, limits


def load(files, output, roi = None, verbose = False):
    """!
    Loads the volume in path and returns it as a 3D array.
    
    Assumes the volume in path is saved as a number of tiff files.
    Can load part of the volume instead of all if select is not None.
    If files is not None, all the other parameters are ignored.
    @param files: The tiff files from which to load the volume (list of strings).
    @param output: Container to put the output in.
    @param roi: (y_min, y_max, x_min, x_max) pixel range to load.
    @param verbose: print (loading %i of %i files)
    @return The read in volume and the description of the first tiff file.
    """
    for f, filename in enumerate(files):
        if len(files) > 1 and verbose:
            printcounting('loading %i of %i files' % (f+1, len(files)), f, len(files))
        try:
            output[f,...] = load_tiff(filename)[0] if roi is None else load_tiff(filename)[0][roi[0]: roi[1], roi[2]: roi[3]]
        except (TiffFileError, UnboundLocalError):
            if f == 0: raise ValueError("Try to solve corrupted files better")
            output[f, ...] =  output[f-1, ...]
    return output



def write_tiff(filename, slc, limits, dtype):
    """Saves a single 2D array as a tiff, linearly rescaling it from `limits` into
    `dtype`'s integer range (clipping out-of-range values) and embedding the slope/offset
    in the tiff description so it can be recovered exactly by `load_and_scale_tiff`/
    `read_description`.
    """
    slope, offset = get_slope_offset(limits, dtype)
    description = get_description(slope, offset)
    slc = np.clip(slc, limits[0], limits[1])
    out = (slc - offset) / slope
    out = out.astype(dtype)
    tifffile.imwrite(filename, out, description = description, compression=False)


def write_tiffs(folder, prefix, volume, limits, dtype):
    """Saves each slice of a 3D volume as a separate tiff file, with the same
    slope/offset rescaling as `write_tiff`.

    NOTE: not currently used/called anywhere in src/ or scripts/.
    """
    slope, offset = get_slope_offset(limits, dtype)
    description = get_description(slope, offset)
    for fi, slc in enumerate(volume):
        out = np.rint((slc - offset) / slope)
        out = np.clip(out, 0, np.iinfo(dtype).max)
        out = out.astype(dtype)
        filename = os.path.join(folder, f'{prefix}_{fi:06d}.tif')
        tifffile.imwrite(filename, out, description = description, compression=False)


def save_projections(sinogram, folder, subfolder ="projections", normalise = True):
    """Writes every projection in `sinogram` to `folder/subfolder` as a tiff. If
    normalise is True, undoes the log-transform (assumes `sinogram` stores -log(I/I0)
    values) before saving as scaled uint16; otherwise writes the raw values directly.
    """
    os.makedirs(os.path.join(folder, subfolder), exist_ok=True)
    for pi in range(sinogram.num_projections):
        filename = os.path.join(folder, subfolder, f'proj_{pi:06d}.tif')
        projection = sinogram[:, pi].squeeze().detach().cpu().numpy()
        if normalise:
            projection = np.exp(-projection)
            write_tiff(filename, projection, limits=(0, 1), dtype=np.uint16)
        else:
            tifffile.imwrite(filename, projection, compression=False)

def save_projections_from_sample(ct_sample, ref_sinogram, save_fmt, sino_params, projection_times, mem_projections,
                                 subset_size, normalise = False):
    """Simulates a sinogram from a sample forward-model (`ct_sample`), subset by subset,
    then saves the result via `save_projections`.

    Args:
        ct_sample: forward-model callable that projects the sample into a sinogram sample
            given the sampled projection indices.
        ref_sinogram: reference sinogram providing num_projections and roi.
        save_fmt: (scan_folder_base, subfolder) tuple passed through to save_projections.
        sino_params: sinogram geometry parameters used to build the simulated CTDatasetSparse.
        projection_times: per-projection timestamps, forwarded to CTDatasetSparse.
        mem_projections: number of projections CTDatasetSparse keeps in memory at once.
        subset_size: number of projections simulated per subset.
        normalise: forwarded to save_projections (whether to undo the log-transform before saving).
    """
    #TODO save subsets - if datasize large
    scan_folder_base, subfolder = save_fmt
    sampler = samplers.OrderedSubsetSampler(range(ref_sinogram.num_projections), subset_size=subset_size)
    simulated_sinogram = dl.CTDatasetSparse(sino_params, projection_times=projection_times, device=ct_sample.device,
                                            mem_projections=mem_projections, roi=ref_sinogram.roi,
                                            fixed_memory=mem_projections - subset_size)

    for si in range(sampler.num_subsets):
        sampled_projections = sampler.next_projections().to(simulated_sinogram.device)
        sinogram_sample = 0 * simulated_sinogram[:, sampled_projections]
        sinogram_sample = ct_sample(sinogram_sample, sampled_projections)
        simulated_sinogram[:, sampled_projections] = sinogram_sample
    save_projections(simulated_sinogram, scan_folder_base, subfolder=subfolder, normalise=normalise)

def save_trajectories(gt_component=None, recon_component=None, folder="./", filename = "trajectories"):
    """Saves the reconstructed (and, if given, ground-truth) particle track control
    points, radii and attenuations to a single .npy file as plain numpy arrays, for later
    analysis outside of torch.
    """
    def convert_params_to_numpy(params):
        tracks, (radii, attenuations) = params
        tracks_np = tracks.detach().cpu().numpy()
        radii_np = radii.detach().cpu().numpy()
        attenuations_np = attenuations.detach().cpu().numpy()
        return tracks_np, (radii_np, attenuations_np)

    os.makedirs(folder, exist_ok=True)
    shape_model = recon_component.shape_model  # type: ShapeModel
    recon_params = recon_component.track_model.control_points, (shape_model.radii, shape_model.attenuations)
    recon_params_np = convert_params_to_numpy(recon_params)
    all_data = {'reconstruction': recon_params_np}
    if gt_component is not None:
        gt_track_model = gt_component.track_model  # type: TrackModel
        gt_shape_model = gt_component.shape_model  # type: ShapeModel
        gt_params = gt_track_model.control_points, (gt_shape_model.radii, gt_shape_model.attenuations)
        gt_params_np = convert_params_to_numpy(gt_params)
        all_data.update({'ground truth': gt_params_np})

    path = os.path.join(folder, filename + '.npy')
    np.save(path, all_data)
    print(f"Saved params to {path}")

def create_reference_scan(scan_folder_full, scan_folder_ref, scan_fmt, io_file, di_file, proj_per_rot,
                          num_rotations, start_rotation = 0, ref_func = np.max):
    """Builds one reference projection per rotation angle by combining `num_rotations`
    repeated acquisitions with `ref_func` (default max) after flat/dark correction - used
    to create a static background reference from a dynamic scan for later subtraction.

    Args:
        proj_per_rot: number of projections per rotation (also the number of reference
            projections produced).
        num_rotations: how many repeated rotations, starting at start_rotation, to combine
            per angle.
        start_rotation: which rotation to start combining from (skips earlier ones).
        ref_func: reduction function applied across rotations at each angle.
    """
    os.makedirs(scan_folder_ref, exist_ok=True)
    start_projection = start_rotation * proj_per_rot
    files = filelist(scan_folder_full, fmt=scan_fmt)[start_projection:start_projection + num_rotations*proj_per_rot]
    flat, _ = load_tiff(io_file)
    dark, _ = load_tiff(di_file)
    height, width = flat.shape
    for i in range(proj_per_rot):
        print(i)
        projections = load(files[i::proj_per_rot], np.zeros((num_rotations, height, width), flat.dtype),
                              None, False)
        projections = projections.astype(np.float32)
        projections = (projections - dark) / (flat - dark)

        ref_projection = ref_func(projections, axis=0)
        write_tiff(os.path.join(scan_folder_ref, f'proj_{i:06d}.tif'),
                      ref_projection, [0, 1.4], np.uint16)

def normalise_and_save_projections(scan_folder_full, save_folder, scan_fmt, io_file, di_file, projection_indices):
    """Flat/dark-corrects the projections at `projection_indices` from scan_folder_full
    and writes them to save_folder as sequentially-numbered tiffs.
    """
    os.makedirs(save_folder, exist_ok=True)
    files = np.asarray(filelist(scan_folder_full, fmt=scan_fmt))
    flat, _ = load_tiff(io_file)
    dark, _ = load_tiff(di_file)
    height, width = flat.shape
    projections = load(files[projection_indices], np.zeros((len(projection_indices), height, width), flat.dtype),
                           None, False)
    projections = projections.astype(np.float32)
    projections = (projections - dark) / (flat - dark)
    for i, proj in enumerate(projections):
        write_tiff(os.path.join(save_folder, f'proj_{i:06d}.tif'),
                   proj, [0, 1.4], np.uint16)




def get_active_branch_name():
    """Returns the name of the currently checked-out git branch by walking up from this
    file to find the repository's .git directory and reading its HEAD file directly
    (avoids depending on the process's cwd or on GitPython for this simple lookup).
    Returns None if no .git directory is found, or if HEAD isn't a symbolic ref.
    """
    # Traverse up parent directories from this file until .git is found - solve cwd issues
    current = Path(__file__).resolve()
    git_dir = next(
        (p / ".git" for p in [current, *current.parents] if (p / ".git").exists()),
        None,
    )

    if not git_dir:
        return None

    head_path = git_dir / "HEAD"
    with head_path.open("r", encoding="utf-8") as f:
        content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]

    return None


def load_simulated_tracks(track_file, device, num_particles, valid_range = None):
    """
    Loads simulated tracks from a dataframe similar to trackpy output into an (N,M,3) torch tensor.
    Assumes columns of particleID, timestep, x,y,z

    @param track_file: The saved dataframe of tracks
    @param device: Device to put the tensor on
    @param num_particles: Number of particles to load
    @param valid_range: min and max range for x,y,z coordinates
    @return The loaded track tensor and the associated timesteps.
    """
    import pandas as pd
    df = pd.read_csv(track_file)

    if valid_range is not None:
        # ((0, pore_mask.shape[2]), (0, pore_mask.shape[1]), (30, pore_mask.shape[0] - 10))
        mask_outside = (
                (df['x'] < valid_range[0][0]) | (df['x'] > valid_range[0][1]) |
                (df['y'] < valid_range[1][0]) | (df['y'] > valid_range[1][1]) |
                (df['z'] < valid_range[2][0]) | (df['z'] > valid_range[2][1])
        )
        bad_particles = df.loc[mask_outside, 'particleID'].unique()
        df = df[~df['particleID'].isin(bad_particles)].copy()

    unique_particles = pd.unique(df['particleID'])
    unique_timesteps = pd.unique(df['timestep'])

    # Create a mapping from ID/index to an integer index for the array
    particle_id_to_idx = {pid: i for i, pid in enumerate(unique_particles)}
    timestep_to_idx = {ts: i for i, ts in enumerate(unique_timesteps)}
    df['particle_idx'] = df['particleID'].map(particle_id_to_idx)
    df['timestep_idx'] = df['timestep'].map(timestep_to_idx)

    # Reshape the data for each coordinate using pivot_table
    x_pivot = df.pivot_table(index='particle_idx', columns='timestep_idx', values='x', fill_value=np.nan)
    y_pivot = df.pivot_table(index='particle_idx', columns='timestep_idx', values='y', fill_value=np.nan)
    z_pivot = df.pivot_table(index='particle_idx', columns='timestep_idx', values='z', fill_value=np.nan)

    x_tensor = torch.from_numpy(x_pivot.values).float()
    y_tensor = torch.from_numpy(y_pivot.values).float()
    z_tensor = torch.from_numpy(z_pivot.values).float()

    reshaped_tensor = torch.stack([x_tensor, y_tensor, z_tensor], dim=-1).to(device)
    timesteps = torch.tensor(unique_timesteps).float().to(device)

    num_total_particles = reshaped_tensor.shape[0]
    indices = torch.randperm(num_total_particles)[:num_particles]
    reshaped_tensor = reshaped_tensor[indices, ...]

    return reshaped_tensor, timesteps


if __name__ == '__main__':
    scan_fmt = '49_inj450nl_%s.tif'
    scan_folder_full = r"J:\Data\Experimental\Parsa\49_inj450nl_Ag_Rubo4mm_n3_20260130"
    save_folder = r"J:\Processing and tracking\CTracks-porous\Newtonian-Parsa\450nlmin\normalised_projections\\"
    projection_indices = np.arange(0, 850, 1, dtype=int)

    io_file = os.path.join(scan_folder_full, "io000001.tif")
    di_file = os.path.join(scan_folder_full, "di000001.tif")

    normalise_and_save_projections(scan_folder_full, save_folder, scan_fmt, io_file, di_file, projection_indices)



def get_version_name():
    """Builds a short run identifier combining the current date/time and a 4-character
    git commit hash, used to tag output/run folders with the code version that produced
    them.
    """
    # Get current date and time
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M")

    # Get current code version as hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    short_sha = repo.git.rev_parse(sha, short=4)
    run_name = f"{date_time}_{short_sha}"
    return run_name
