import os
import glob
import time
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
from utils.processing_transformation import px2cm, \
    normalize_origin_of_compartment
from utils.processing_methods import distance_to_wall_chunk
from utils.metrics import update_filter_three_points, compute_turning_angles, \
    compute_step_lengths

from config import BLOCK, BACK, BATCH_SIZE, FRAMES_PER_SECOND
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
    data, new_area = normalize_origin_of_compartment(data, area, BACK in fk)
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
    # check if X is finite
    if not np.all(np.isfinite(X)):
        raise ValueError("Not all values in X are finite, \
            something went wrong in the feature computation", X)
    return X, new_area


def compute_projections(fish_key, day, area_tuple, excluded_days=dict()):
    cam, pos = fish_key.split("_")
    is_back = pos == BACK
    keys, data_in_batches = csv_of_the_day(
        cam,
        day,
        is_back=is_back,
        batch_keys_remove=excluded_days.get(f"{BLOCK}_{fish_key}_{day}",[])
    )
    if len(data_in_batches) == 0:
        print(f"{fish_key} for day {day} is empty! ")
        return None, None
    daytime_DF = start_time_of_day_to_seconds(day.split("_")[1])\
        * FRAMES_PER_SECOND
    for k, df in zip(keys, data_in_batches):
        df.index = df.FRAME+(int(k)*BATCH_SIZE)+daytime_DF
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
    trimmed=False
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
    print(f"{fk} {day} {X.shape}")
    if X.shape[0] < 1000:
        print("Skip: number of datapoints to small")
        return None
    if trimmed:
        hdf5storage.write(
            data={
                'projections': X[:, 1:4],
                'positions': X[:, 4:],
                'df_time_index': X[:, 0],
            },
            path='/',
            truncate_existing=True,
            filename=filename,
            store_python_metadata=False,
            matlab_compatible=True
        )
    else:
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
    trimmed=False
):
    area_f = get_area_functions()
    if fish_keys is None:
        fish_keys = get_camera_pos_keys()
    numProcessors = mp.cpu_count()
    for i, fk in enumerate(fish_keys):
        t1 = time.time()
        pool = mp.Pool(numProcessors)
        days = get_days_in_order(
            camera=fk.split("_")[0],
            is_back=fk.split("_")[1] == BACK
        )
        if trimmed:
            trimmed_directory = projectPath + '/Projections_trimmed'
            if not os.path.exists(trimmed_directory):
                os.makedirs(trimmed_directory, exist_ok=True)
                print('trimmed projections directory created')
            outs = pool.starmap(compute_and_write_projection, [(
                    fk, day, (fk, area_f(fk)),
                    projectPath + f'/Projections_trimmed/{BLOCK}_{fk}_{day}\
                        _pcaModes.mat',
                    recompute, excluded_days, trimmed
                ) for day in days]
            )
        else:
            outs = pool.starmap(compute_and_write_projection, [(
                fk, day, (fk, area_f(fk)), 
                projectPath + f'/Projections/{BLOCK}_{fk}_{day}_pcaModes.mat',
                recompute, excluded_days) for day in days]
            )
        pool.close()
        pool.join()
        print('\t Processed fish #%4i %s out of %4i in %0.02fseconds.\n'%(i+1, fk, len(fish_keys), time.time()-t1))


def compute_all_projections_filtered(parameters, trimmed = False):
    fish_keys = get_camera_pos_keys() # get all fish keys
    for key in block1_remove:
        if BLOCK in key:
            fk = "_".join(key.split("_")[1:])
            if fk in fish_keys:
                fish_keys.remove(fk)
    excluded = get_excluded_days(list(map(lambda f: f"{BLOCK}_{f}", fish_keys)))
    compute_all_projections(
        parameters.projectPath,
        fish_keys,
        excluded_days=excluded,
        recompute=True,
        trimmed=trimmed
    )


def load_trajectory_data(parameters, fk="", day=""):
    data_by_day = []
    pfile = glob.glob(
        parameters.projectPath+f'/Projections/{fk}*_{day}*_pcaModes.mat'
    )
    pfile.sort()
    print('loading trajectory data')
    for f in tqdm(pfile): 
        data = hdf5storage.loadmat(f)
        data_by_day.append(data)
    return data_by_day


if __name__ == "__main__":
    BLOCK = 'block1'
    os.environ['BLOCK'] = 'block1'
    print("Start computation for: ", BLOCK)
    parameters = set_parameters()
    compute_all_projections_filtered(parameters, trimmed=False)

    BLOCK = 'block2'
    os.environ['BLOCK'] = 'block2'
    print("Start computation for: ", BLOCK)
    parameters = set_parameters()
    compute_all_projections_filtered(parameters, trimmed=False)