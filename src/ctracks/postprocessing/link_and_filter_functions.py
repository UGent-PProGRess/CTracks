import numpy as np
import torch
import pandas as pd
import trackpy as tp
from byotrack.api.detector.detections import Detections
from byotrack.implementation.linker.frame_by_frame.kalman_linker import KalmanLinker, KalmanLinkerParameters
from laptrack import LapTrack
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def localVelocity(trajectories):
    """Compute per-particle velocities (finite differences over time, accounting for frame gaps)
    from a linked-trajectory dataframe (as produced by trackpy-style linkers).

    Args:
        trajectories: dataframe with columns 'particle', 'frame', 'x', 'y', 'z', and optionally
            'radius'/'attenuation' (shape params carried along per detection).

    Returns:
        velocities, velocityPos, velocityMags, particleID, particleTime, velocityShape
        (velocityShape is None if no shape columns were present). All arrays are prefixed with
        one dummy zero row from the accumulation loop below.
    """
    velocityPos = np.zeros((1, 3))
    velocities = np.zeros((1, 3))
    velocityMags = np.zeros((1, 1))
    particleID = np.zeros((1, 1))
    particleTime = np.zeros((1, 1))
    particles = trajectories.particle.unique()

    if "radius" in trajectories.columns: velocityShape = np.zeros((1,2))
    else: velocityShape = None


    for particle in particles:
        xlocs = trajectories[trajectories.particle == particle]["x"].values
        ylocs = trajectories[trajectories.particle == particle]["y"].values
        zlocs = trajectories[trajectories.particle == particle]["z"].values
        locs = np.stack(np.transpose((zlocs, ylocs, xlocs)))

        ts = trajectories[trajectories.particle == particle]["frame"].values

        partVelocities = np.gradient(locs, ts, axis=0)  # accounting for timestamp jumps - memory
        partVelocityPos = locs

        if velocityShape is not None:
            rad = trajectories[trajectories.particle == particle]["radius"].values
            att = trajectories[trajectories.particle == particle]["attenuation"].values
            partVelocityShape = np.stack((rad, att), axis=1)

        if (partVelocityPos.size > 0):
            velocityPos = np.append(velocityPos, partVelocityPos, axis=0)
            velocities = np.append(velocities, partVelocities, axis=0)
            particleID = np.append(particleID, np.ones(len(partVelocityPos)) * particle)
            particleTime = np.append(particleTime, ts)
            if velocityShape is not None:
                velocityShape = np.append(velocityShape, partVelocityShape, axis=0)
    velocityMags = np.sqrt((velocities ** 2).sum(axis=1))
    return velocities, velocityPos, velocityMags, particleID, particleTime, velocityShape

def link_ctrack(ctrack_pos, dim, trajectoryMinLength = 4, trajectoryMemory=0, trajectorySearchRange=6, shape_params = None, init_vel = None):
    """Link per-frame detections into trajectories using trackpy's velocity-predicting linker.

    Args:
        ctrack_pos: list of per-frame detection arrays, each (N, 3) in (z, y, x).
        dim: (z, y, x) volume shape, used to seed corner initial-guess positions for the predictor.
        init_vel: currently unused (kept for interface compatibility with callers that pass it).

    Returns:
        Same as ``filter_and_calc_vel``: (velocityData dataframe, frame-list conversion tuple).
    """
    trajectoryAdaptiveStep = 0.95
    trajectoryAdaptiveStop = 2
    velocityPredictInit = [1, 0, 0]
    velocityPredictSpan = 3


    #convert to dataframe
    f = convert_frame_lists_to_trackpy(ctrack_pos, shape_params)

    #run trackpy linking
    f_masked_split = [pd.DataFrame(y) for x, y in f.groupby('frame', as_index=False)]
    zDim = dim[0]
    yDim = dim[1]
    xDim = dim[2]

    initial_guess_positions = [[0, 0, 0], [zDim, 0, 0], [0, yDim, 0], [0, 0, xDim], [zDim, yDim, 0], [0, yDim, xDim],
                                 [zDim, 0, xDim], [zDim, yDim, xDim]]
    initial_guess_velocs = [velocityPredictInit, velocityPredictInit, velocityPredictInit, velocityPredictInit,
                            velocityPredictInit, velocityPredictInit, velocityPredictInit, velocityPredictInit, ]

    pred = tp.predict.NearestVelocityPredict(
        initial_guess_positions= initial_guess_positions, pos_columns=["z", "y", "x"],
        initial_guess_vels=initial_guess_velocs,
        span=velocityPredictSpan)
    trajectories = pd.concat(
        pred.link_df_iter(f_masked_split, pos_columns=["z", "y", "x"], memory=trajectoryMemory,
                          search_range=trajectorySearchRange, adaptive_step=trajectoryAdaptiveStep,
                          adaptive_stop=trajectoryAdaptiveStop))
    return filter_and_calc_vel(trajectories, trajectoryMinLength)


def convert_trackpy_to_frame_lists(df, crop_transform = np.zeros(3), frame_range = None):
    """Convert a trackpy-style dataframe (columns x, y, z, vx, vy, vz, frame) back into
    per-frame position/velocity lists, plus flattened concatenations across all frames.

    Args:
        crop_transform: offset added to positions (e.g. to undo an ROI crop).
        frame_range: explicit frame values to iterate, defaults to the dataframe's unique frames.

    Returns:
        pos, vel: lists of per-frame (N, 3) arrays.
        pos_all, vel_all: (sum_N, 3) arrays concatenated across frames.
        vel_all_magnitude: (sum_N,) velocity magnitudes.
    """
    pos, vel = [], []  # frames,<N>,3
    loop_range = df.frame.unique() if frame_range is None else frame_range

    for frame in loop_range:
        frame_data = df[df['frame'] == frame]
        pos.append(frame_data[['x', 'y', 'z']].values+crop_transform)
        vel.append(frame_data[['vx', 'vy', 'vz']].values)
    pos_all = np.concatenate(pos, axis=0)
    vel_all = np.concatenate(vel, axis=0)
    vel_all_magnitude = np.linalg.norm(vel_all, axis=1)
    return pos, vel, pos_all, vel_all, vel_all_magnitude

def convert_frame_lists_to_trackpy(frame_list, shape_params = None, vel_params = None):
    """Convert a list of per-frame detection arrays (each (N, 3) in (z, y, x)) into a single
    trackpy-style dataframe with columns frame, z, y, x, and optionally radius/attenuation
    (from ``shape_params``) and vz/vy/vx (from ``vel_params``)."""
    all_frame_rows = []

    for frame_number, frame_detections in enumerate(frame_list):
        frame_detections = np.asarray(frame_detections)
        if frame_detections.size > 0:
            num_detections = frame_detections.shape[0]
            reordered_detections = frame_detections[:, [2, 1, 0]]
            frame_column = np.full((num_detections, 1), frame_number, dtype=int)
            row_data = [frame_column, reordered_detections]
            if shape_params is not None:
                rad = np.asarray(shape_params[0][frame_number])[:, np.newaxis]
                att = np.asarray(shape_params[1][frame_number])[:, np.newaxis]
                row_data.append(rad)
                row_data.append(att)
            if vel_params is not None:
                frame_vel = vel_params[frame_number][:, [2,1,0]]
                row_data.append(frame_vel)

            frame_rows = np.hstack(row_data)
            all_frame_rows.append(frame_rows)

    cols = ['frame', 'z', 'y', 'x']
    if shape_params is not None: cols += ['radius', 'attenuation']
    if vel_params is not None: cols += ['vz', 'vy', 'vx']

    if all_frame_rows:
        final_array = np.vstack(all_frame_rows)
        df = pd.DataFrame(final_array,columns=cols)
        df['frame'] = df['frame'].astype(int)
    else:
        # Return an empty DataFrame with the correct columns if no detections were found
        df = pd.DataFrame(columns=cols)
        df['frame'] = df['frame'].astype(int)  # Set column type even for empty DF

    return df

def lap_track(data, trajectoryMinLength=5, trajectoryMemory=1, trajectorySearchRange=60, memory_range = 100):
    """Link per-frame detections into trajectories using LapTrack (linear assignment problem
    based linking, with gap closing but no splitting/merging)."""
    print("Linking tracks with LAP linker")
    df = convert_frame_lists_to_trackpy(data)
    tracker = LapTrack(
        track_dist_metric="sqeuclidean",  # Evaluates spatial proximity using squared distance
        track_cost_cutoff= trajectorySearchRange ** 2,  # Max squared distance for frame-to-frame linking
        gap_closing_cost_cutoff= memory_range ** 2,  # Max squared distance allowed to close a temporal gap
        gap_closing_max_frame_count=trajectoryMemory,  # Maximum number of frames a particle can be missing
        splitting_cost_cutoff=False,  # Set to False to disable track splitting
        merging_cost_cutoff=False  # Set to False to disable track merging
    )

    track_df, _, _ = tracker.predict_dataframe(
        df,
        coordinate_cols=['x', 'y', 'z'],  # Tell the tracker which columns are spatial
        frame_col='frame'  # Identify the time dimension
    )
    # Resetting the index pulls 'frame' back out as a standard column if it was hidden in the index
    if 'frame' not in track_df.columns:
        track_df = track_df.reset_index()

    # Rename the default 'track_id' to 'particle'
    track_df = track_df.rename(columns={'track_id': 'particle'})

    # Filter and reorder the DataFrame to match your desired output
    final_tracks = track_df[['frame', 'particle', 'x', 'y', 'z']]

    return filter_and_calc_vel(final_tracks, trajectoryMinLength)


def kalman_track(list_data, trajectoryMinLength=6, trajectoryMemory=0, trajectorySearchRange=20, detection_confidence = 2, model_confidence = 1):
    """Link per-frame detections into trajectories using a byotrack constant-velocity Kalman linker."""
    print("Linking tracks with Kalman linker")
    data = convert_frame_lists_to_trackpy(list_data)

    # 2. Convert Pandas DataFrame to a list of ByoTrack Detections
    frames = range(int(data['frame'].max()) + 1)
    detections_sequence = []

    for f in frames:
        df_frame = data[data['frame'] == f]
        if df_frame.empty:
            # Append empty detections for missing frames
            empty_tensor = torch.empty((0, 3), dtype=torch.float32)
            detections_sequence.append(Detections(data={"position": empty_tensor}, frame_id=f))
        else:
            # Extract z, y, x as a PyTorch tensor
            positions = torch.tensor(df_frame[['z', 'y', 'x']].values, dtype=torch.float32)
            detections_sequence.append(Detections(data={"position": positions}, frame_id=f))

    # 3. Initialize the Kalman Linker Parameters
    parameters = KalmanLinkerParameters(
        kalman_order=1,  # 1 = Constant Velocity Model
        association_threshold= trajectorySearchRange,  # Radius to search around the PREDICTED future position
        n_gap=trajectoryMemory,  # Allow a particle to be missing for 2 frames
        detection_std=detection_confidence,
        process_std=model_confidence
    )

    # 4. Run the Tracker
    linker = KalmanLinker(parameters)
    dummy_video = [np.empty((1, 1, 1, 1), dtype=np.uint8)] * len(detections_sequence)
    tracks = linker.run(dummy_video, detections_sequence)

    # 5. Convert ByoTrack output back to a Pandas DataFrame
    output_data =[]
    for track in tracks:
        for t_idx, point in enumerate(track.points):
            if not torch.isnan(point).any():
                z, y, x = point.tolist()
                frame = track.start + t_idx
                output_data.append({
                    'frame': frame,
                    'particle': track.identifier,
                    'x': x, 'y': y, 'z': z
                })

    track_df = pd.DataFrame(output_data)
    track_df = track_df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
    # Filter and reorder the DataFrame to match your desired output
    final_tracks = track_df[['frame', 'particle', 'x', 'y', 'z']]

    return filter_and_calc_vel(final_tracks, trajectoryMinLength)


def lap_track_with_velocity(list_data, list_vel_data, trajectoryMinLength=6, trajectorySearchRange=20, trajectoryMemory=2,
                            max_history_weight=0.8, shape_params = None):
    """Custom frame-by-frame linker: predicts each active track's next position from a weighted
    blend of the detector-reported velocity and the track's own recent history (the history
    weight grows with track length, up to ``max_history_weight``), assigns detections to
    predictions via linear sum assignment on squared distance, allows tracks to survive up to
    ``trajectoryMemory`` missed frames, and linearly interpolates positions across any gaps
    before returning the finished tracks.
    """
    print("Tracking with dynamic history weighting, memory, and interpolation...")

    df = convert_frame_lists_to_trackpy(list_data, vel_params=list_vel_data, shape_params = shape_params)
    frames = sorted(df['frame'].unique())

    active_tracks = {}  # Stores currently alive tracks
    missed_frames = {}  # Tracks how many consecutive frames a track has been missing
    finished_tracks = []  # Stores tracks that have ended
    next_track_id = 0

    for f in frames:
        current_detections = df[df['frame'] == f].to_dict('records')

        # If no active tracks, initialize all detections as new tracks
        if not active_tracks:
            for det in current_detections:
                active_tracks[next_track_id] = [det]
                missed_frames[next_track_id] = 0
                next_track_id += 1
            continue

        track_ids = list(active_tracks.keys())
        predicted_positions = []

        # 1. PREDICT PHASE: Calculate hybrid prediction for each active track
        for tid in track_ids:
            track = active_tracks[tid]
            last_det = track[-1]

            pos = np.array([last_det['x'], last_det['y'], last_det['z']])
            det_vel = np.array([last_det['vx'], last_det['vy'], last_det['vz']])

            dt = f - last_det['frame']
            track_length = len(track)

            # --- DYNAMIC WEIGHTING LOGIC ---
            if track_length < 2:
                w_history = 0.0
                hist_vel = np.zeros(3)
            else:
                prev_det = track[-2]
                prev_pos = np.array([prev_det['x'], prev_det['y'], prev_det['z']])

                dt_hist = last_det['frame'] - prev_det['frame']
                hist_vel = (pos - prev_pos) / dt_hist if dt_hist > 0 else np.zeros(3)

                w_history = min(max_history_weight, (track_length - 1) * 0.2)

            w_detector = 1.0 - w_history
            hybrid_vel = (w_detector * det_vel) + (w_history * hist_vel)

            predicted_pos = pos + (hybrid_vel * dt)
            predicted_positions.append(predicted_pos)

        # 2. COST MATRIX: Distance between predicted positions and actual detections
        predicted_positions = np.array(predicted_positions)
        actual_positions = np.array([[d['x'], d['y'], d['z']] for d in current_detections])

        if len(actual_positions) > 0:
            cost_matrix = cdist(predicted_positions, actual_positions, metric='sqeuclidean')

            # --- COST MATRIX GATING ---
            max_allowed_cost = trajectorySearchRange ** 2
            cost_matrix[cost_matrix > max_allowed_cost] = 1e9

            row_inds, col_inds = linear_sum_assignment(cost_matrix)

            assigned_tracks = set()
            assigned_detections = set()

            # 3. UPDATE PHASE
            for r, c in zip(row_inds, col_inds):
                if np.sqrt(cost_matrix[r, c]) <= trajectorySearchRange:
                    tid = track_ids[r]
                    active_tracks[tid].append(current_detections[c])
                    missed_frames[tid] = 0
                    assigned_tracks.add(tid)
                    assigned_detections.add(c)
        else:
            assigned_tracks = set()
            assigned_detections = set()

        # 4. CLEANUP PHASE
        unassigned_tracks = set(track_ids) - assigned_tracks
        for tid in unassigned_tracks:
            missed_frames[tid] += 1
            if missed_frames[tid] > trajectoryMemory:
                # The track has exceeded its memory limit; remove it from active
                track_data = active_tracks.pop(tid)
                del missed_frames[tid]

                # only keep tracks that span more than one detection
                if len(track_data) > 1:
                    finished_tracks.append({'particle': tid, 'data': track_data})

        unassigned_detections = set(range(len(current_detections))) - assigned_detections
        for c in unassigned_detections:
            active_tracks[next_track_id] = [current_detections[c]]
            missed_frames[next_track_id] = 0
            next_track_id += 1

    for tid, data in active_tracks.items():
        finished_tracks.append({'particle': tid, 'data': data})

    # Linearly interpolate positions across any missed frames within each finished track
    output_rows = []
    for track in finished_tracks:
        tid = track['particle']
        data = track['data']
        if not data: continue

        prev_point = data[0]
        output_rows.append({
            'frame': prev_point['frame'], 'particle': tid,
            'x': prev_point['x'], 'y': prev_point['y'], 'z': prev_point['z']
        })

        for i in range(1, len(data)):
            curr_point = data[i]
            frame_diff = curr_point['frame'] - prev_point['frame']

            # If there is a gap (e.g., missed detections), interpolate the missing frames
            if frame_diff > 1:
                for step in range(1, frame_diff):
                    interp_frame = prev_point['frame'] + step
                    fraction = step / frame_diff

                    interp_x = prev_point['x'] + (curr_point['x'] - prev_point['x']) * fraction
                    interp_y = prev_point['y'] + (curr_point['y'] - prev_point['y']) * fraction
                    interp_z = prev_point['z'] + (curr_point['z'] - prev_point['z']) * fraction

                    output_rows.append({
                        'frame': interp_frame, 'particle': tid,
                        'x': interp_x, 'y': interp_y, 'z': interp_z
                    })

            output_rows.append({
                'frame': curr_point['frame'], 'particle': tid,
                'x': curr_point['x'], 'y': curr_point['y'], 'z': curr_point['z']
            })
            prev_point = curr_point

    result_dataframe = pd.DataFrame(output_rows).sort_values(['particle', 'frame']).reset_index(drop=True)
    final_tracks = result_dataframe[['frame', 'particle', 'x', 'y', 'z']]

    return filter_and_calc_vel(final_tracks, trajectoryMinLength)

def filter_vectors_direct(
            list_data,
            list_vel_data,
            shape_params,
            min_radius=5,  # A%: Drop bottom 5% thinnest particles
            min_atten=5,  # B%: Drop bottom 5% darkest particles
            velocity_maxima = (10,10,10), #the maximum allowed velocities in reconstruction step
            velocity_clip=1.0,  # remove particles within x% of maxima
            k_neighbors=15,  # Number of neighbors for consistency check
            uod_threshold=None,  # Universal Outlier Detection threshold
            eps=0.1,  # Epsilon to prevent division by zero in UOD
            main_flow_axis = 'z',
            threshold_reverse_vel = None,
            isolation_search_radius = None,
            isolation_min_neighbors = 1,
            isolation_per_frame = True
    ):
    """Filtering pipeline used by all analyse_*.py scripts: percentile radius/attenuation cutoffs,
    velocity clipping, then optional UOD (neighborhood consistency), bulk-flow, and isolation
    filters. Returns the filtered dataframe together with its frame-list conversion.
    """
    df = convert_frame_lists_to_trackpy(list_data, vel_params=list_vel_data, shape_params=shape_params)
    df_filtered = df.copy()

    # 1. Filter out the bottom A% thinnest particles
    if min_radius != 0:
        r_thresh = np.percentile(df_filtered['radius'], min_radius)
        if r_thresh != df_filtered['radius'].max():  #check it won't remove everything
            df_filtered = df_filtered[df_filtered['radius'] > r_thresh]

    # 2. Filter out the bottom B% darkest particles
    if min_atten != 0:
        a_thresh = np.percentile(df_filtered['attenuation'], min_atten)
        df_filtered = df_filtered[df_filtered['attenuation'] >= a_thresh]


    if velocity_clip != 0:
        axes = ['vx', 'vy', 'vz']
        for i, vel_max in enumerate(velocity_maxima):
            upper_limit = vel_max * (1-velocity_clip/100)
            lower_limit = -1 * vel_max * (1-velocity_clip/100)
            df_filtered = df_filtered[(df_filtered[axes[i]] > lower_limit) & (df_filtered[axes[i]] < upper_limit)]

    if uod_threshold is not None:
        df_filtered = uod_filter(df_filtered, eps, k_neighbors, uod_threshold)

    if threshold_reverse_vel is not None:
        df_filtered = bulk_flow_filter(df_filtered, main_flow_axis = main_flow_axis, threshold_reverse_vel=threshold_reverse_vel)

    if isolation_search_radius is not None:
        df_filtered = isolation_filter(df_filtered, search_radius = isolation_search_radius, min_neighbors = isolation_min_neighbors, per_frame = isolation_per_frame)

    return df_filtered, convert_trackpy_to_frame_lists(df_filtered)


def uod_filter(df_filtered, eps, k_neighbors, uod_threshold):
    """Universal Outlier Detection: drop particles whose velocity deviates from their spatial
    k-nearest-neighbor median velocity by more than ``uod_threshold`` times the local neighborhood's
    median absolute deviation (processed frame by frame so time steps don't mix)."""
    frames = df_filtered['frame'].unique()
    valid_indices = []
    for f in frames:
        df_frame = df_filtered[df_filtered['frame'] == f]

        # If there aren't enough particles to form a neighborhood, skip or keep
        if len(df_frame) < k_neighbors + 1:
            valid_indices.extend(df_frame.index.tolist())
            continue

        coords = df_frame[['x', 'y', 'z']].values
        vels = df_frame[['vx', 'vy', 'vz']].values

        # Build spatial tree for fast neighbor lookup
        tree = cKDTree(coords)
        # k+1 because the first match is always the point itself (distance 0)
        _, indices = tree.query(coords, k=k_neighbors + 1)

        # Isolate the neighbors (exclude the point itself)
        neighbor_indices = indices[:, 1:]

        # Shape: (N_particles, k_neighbors, 3 dimensions)
        neighbor_vels = vels[neighbor_indices]

        # Calculate the median velocity of the neighborhood for x, y, z
        median_vels = np.median(neighbor_vels, axis=1)

        # Calculate residual of the particle to its neighborhood median
        residuals = np.abs(vels - median_vels)

        # Calculate the median absolute deviation (MAD) of the neighbors
        # This tells us how chaotic the local flow is.
        neighbor_residuals = np.abs(neighbor_vels - median_vels[:, np.newaxis, :])
        median_neighbor_residuals = np.median(neighbor_residuals, axis=1)

        # Normalize the residual by the local chaos (MAD)
        # eps prevents dividing by zero in perfectly uniform flow regions
        normalized_residuals = residuals / (median_neighbor_residuals + eps)

        # A particle is valid if its normalized residual is below the threshold in all 3 axes
        is_valid = np.all(normalized_residuals < uod_threshold, axis=1)

        valid_indices.extend(df_frame.index[is_valid].tolist())
    df_filtered = df_filtered.loc[valid_indices].reset_index(drop=True)
    return df_filtered


# --- Optional Physics-Informed Extra Filters ---

def bulk_flow_filter(df, main_flow_axis='z', threshold_reverse_vel=-2.0):
    """
    Removes particles moving strongly against the bulk pressure gradient.
    Example: If fluid is forced in positive X, discard vx < -2.0
    """
    return df[df[f'v{main_flow_axis}'] >= threshold_reverse_vel]


def isolation_filter(df, search_radius=10.0, min_neighbors=1, per_frame=True):
    """
    Removes particles that are spatially isolated.

    per_frame=True: Only considers neighbors within the same time step.
    per_frame=False: Considers neighbors across all time (useful for finding persistent tracks).
    """
    if per_frame:
        valid_indices = []
        # Grouping by frame is more efficient than unique() + manual slicing
        for _, df_frame in df.groupby('frame'):
            coords = df_frame[['x', 'y', 'z']].values
            tree = cKDTree(coords)

            # count_neighbors: r is radius.
            # We subtract 1 because query_ball_point includes the point itself.
            counts = tree.query_ball_point(coords, r=search_radius, return_length=True) - 1

            is_valid = counts >= min_neighbors
            valid_indices.extend(df_frame.index[is_valid])

        return df.loc[valid_indices]

    else:
        # All frames at once
        coords = df[['x', 'y', 'z']].values
        tree = cKDTree(coords)

        # Again, subtract 1 to exclude self
        counts = tree.query_ball_point(coords, r=search_radius, return_length=True) - 1

        return df[counts >= min_neighbors]

def filter_and_calc_vel(track_df, trajectoryMinLength):
    """Drop trajectories shorter than ``trajectoryMinLength`` (trackpy's ``filter_stubs``), compute
    per-particle velocities, and return both the velocity dataframe and its frame-list conversion."""
    trajectories_filtered = tp.filter_stubs(track_df, trajectoryMinLength)
    vels, velocityPos, velMags, particleVelIDs, velTimes, velocityShape = localVelocity(trajectories_filtered)
    data_dict = {"particle": particleVelIDs.astype(int), "frame": velTimes.astype(int), "z": velocityPos[:, 0],
                 "y": velocityPos[:, 1], "x": velocityPos[:, 2], "vz": vels[:, 0], "vy": vels[:, 1], "vx": vels[:, 2],
                 "velMags": velMags}
    if velocityShape is not None:
        data_dict['radius'] = velocityShape[:, 0]
        data_dict['attenuation'] = velocityShape[:, 1]
    velocityData = pd.DataFrame(data_dict)
    velocityData.drop(index=0, inplace=True)
    # convert back to arrays
    return velocityData, convert_trackpy_to_frame_lists(velocityData)
