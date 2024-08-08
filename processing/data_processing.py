import os
import glob
import pandas as pd
import multiprocessing as mp
import numpy as np
import hdf5storage
from tqdm import tqdm

from utils.utils import set_parameters
from utils.processing_utils import get_camera_pos_keys, csv_of_the_day, \
    start_time_of_day_to_seconds, get_days_in_order
from utils.excluded_days import get_excluded_days, block1_remove
from utils.tank_area_config import get_area_functions
from utils.error_filter import all_error_filters
from processing.processing_transformation import px2cm, \
    normalize_origin_of_compartment
from processing.processing_methods import distance_to_wall_chunk
from utils.metrics import update_filter_three_points, compute_turning_angles, \
    compute_step_lengths

import config
WAVELET = 'wavelet'
clusterStr = 'clusters'


def transform_to_traces_high_dim(data, frame_idx, filter_index, area_tuple):
    """
    Transform the xy data into the feature space
    :param data: xy data
    :param frame_idx: frame index
    :param filter_index: filter index
    :param area_tuple: area tuple
    :return: transformed data [frame, steps, turning angle,
        distance to the wall, x, y] and new area
    """
    fk, area = area_tuple
    data, new_area = normalize_origin_of_compartment(
        data, area, config.BACK in fk
    )
    steps = px2cm(compute_step_lengths(data))
    t_a = compute_turning_angles(data)
    wall = px2cm(distance_to_wall_chunk(data, new_area))
    f3 = update_filter_three_points(steps, filter_index)
    X = np.array((
            frame_idx[1:-1],
            steps[1:],
            t_a,
            wall[1:-1],
            data[1:-1, 0],
            data[1:-1, 1]
        )).T
    X = X[~f3]
    if not np.all(np.isfinite(X)):
        raise ValueError("Not all values in X are finite, \
            something went wrong in the feature computation", X)
    return X, new_area


def compute_projections(fish_key, day, area_tuple, excluded_days=dict()):
    """
    Computes projections for a specific fish, day, and area.
    This function first determines the starting time for the analysis.
    It then filters the dataset to exclude records outside the specified area,
    as well as those containing dirt or NaN (Not a Number) values.
    After filtering, it calculates various features of the fish's movement
    (steps, turning angle, distance to the wall)

    Parameters:
    - fish_key (str): The fish key.
    - day (str): The day.
    - area_tuple (tuple): The area tuple.
    - excluded_days (dict): Dictionary of excluded days.

    Returns:
    - X (numpy.ndarray): The computed projections.
    - new_area (tuple): The new area tuple.
    """
    cam, pos = fish_key.split("_")
    is_back = pos == config.BACK
    keys, data_in_batches = csv_of_the_day(
        cam,
        day,
        is_back=is_back,
        batch_keys_remove=excluded_days.get(
            f"{config.BLOCK}_{fish_key}_{day}", []
        )
    )
    if len(data_in_batches) == 0:
        return None, None
    daytime_DF = start_time_of_day_to_seconds(day.split("_")[1])\
        * config.FRAMES_PER_SECOND
    for k, df in zip(keys, data_in_batches):
        df.index = df.FRAME+(int(k)*config.BATCH_SIZE)+daytime_DF
    data = pd.concat(data_in_batches)
    data_px = data[["xpx", "ypx"]].to_numpy()
    filter_index = all_error_filters(
        data_px,
        area_tuple,
        fish_key=fish_key,
        day=day
    )
    X, new_area = transform_to_traces_high_dim(
        data_px,
        data.index,
        filter_index,
        area_tuple
    )
    return X, new_area


def compute_and_write_projection(
    fk,
    day,
    area_tuple,
    filename,
    recompute=False,
    excluded_days=dict(),
):
    if not recompute and os.path.exists(filename):
        return None
    X, new_area = compute_projections(
        fk,
        day,
        area_tuple,
        excluded_days=excluded_days
    )
    if X is None:
        return None
    if X.shape[0] < 1000:
        return None
    hdf5storage.write(
        data={
            'projections': X[:, 1:4],
            'positions': X[:, 4:],
            'area': new_area,
            'df_time_index': X[:, 0],
            'day': day,
            'fish_key': fk
        },
        path='/',
        truncate_existing=True,
        filename=filename,
        store_python_metadata=False,
        matlab_compatible=True
    )
    return None


def compute_all_projections(
    projectPath,
    fish_keys=None,
    recompute=False,
    excluded_days=dict(),
):
    area_f = get_area_functions()
    if fish_keys is None:
        fish_keys = get_camera_pos_keys()
    numProcessors = mp.cpu_count()
    for i, fk in tqdm(enumerate(fish_keys), total=len(fish_keys)):
        pool = mp.Pool(numProcessors)
        days = get_days_in_order(
            camera=fk.split("_")[0],
            is_back=fk.split("_")[1] == config.BACK
        )
        _ = pool.starmap(compute_and_write_projection, [(
            fk, day, (fk, area_f(fk)),
            projectPath +
            f'/Projections/{config.BLOCK}_{fk}_{day}_pcaModes.mat',
            recompute, excluded_days) for day in days]
        )
        pool.close()
        pool.join()


def compute_all_projections_filtered(parameters):
    fish_keys = get_camera_pos_keys()
    for key in block1_remove:
        if config.BLOCK in key:
            fk = "_".join(key.split("_")[1:])
            if fk in fish_keys:
                fish_keys.remove(fk)
    excluded = get_excluded_days(
        list(map(lambda f: f"{config.BLOCK}_{f}", fish_keys))
    )
    compute_all_projections(
        parameters.projectPath,
        fish_keys,
        excluded_days=excluded,
        recompute=True,
    )


def load_trajectory_data(parameters, fk="", day=""):
    data_by_day = []
    pfile = glob.glob(
        parameters.projectPath+f'/Projections/{fk}*_{day}*_pcaModes.mat'
    )
    pfile.sort()
    for f in tqdm(pfile):
        data = hdf5storage.loadmat(f)
        data_by_day.append(data)
    return data_by_day


if __name__ == "__main__":
    for block_nr in range(1, config.N_BLOCKS+1):
        exec(f"config.BLOCK = config.BLOCK{block_nr}")
        exec(f"os.environ['BLOCK'] = config.BLOCK{block_nr}")
        config.DIR_CSV_LOCAL = f"{config.PROJ_PATH}/FE_tracks_060000_"\
            f"{config.BLOCK}"
        config.err_file = f"{config.DIR_CSV_LOCAL}/results/log_error.csv"
        config.area_back = f"{config.DIR_CSV_LOCAL}/area_config/areas_back"
        config.area_front = f"{config.DIR_CSV_LOCAL}/area_config/areas_front"
        config.config_path = f"{config.DIR_CSV_LOCAL}/{config.CONFIG_DATA}"

        print("Start computation for: ", config.BLOCK)
        parameters = set_parameters()
        compute_all_projections_filtered(parameters)
