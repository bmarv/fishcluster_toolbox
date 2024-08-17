import math
import pandas as pd
import numpy as np
import re
import os
import glob
from itertools import product
import config


def get_directory(is_back=None):
    BLOCK = config.BLOCK
    if is_back is None:
        raise Exception("define kwargs is_back")
    if is_back:
        return f'{config.PROJ_PATH}/PE_tracks_final_{BLOCK}'\
            f'/PE_tracks_final_{BLOCK}_back'
    else:
        return f'{config.PROJ_PATH}/PE_tracks_final_{BLOCK}'\
            f'/PE_tracks_final_{BLOCK}_front'


def get_camera_names(is_back=False):
    dir_ = get_directory(is_back)
    return sorted(
        [
            name for name in os.listdir(dir_)
            if len(name) == 8 and name.isnumeric()
        ]
    )


def get_fish2camera_map():
    l_front = list(
        product(
            get_camera_names(is_back=config.BACK == config.FRONT),
            [config.FRONT]
        )
    )
    l_back = list(
        product(
            get_camera_names(is_back=config.BACK == config.BACK),
            [config.BACK]
        )
    )
    return np.array(l_back + l_front)


def get_camera_pos_keys():
    m = get_fish2camera_map()
    return ["%s_%s" % (c, p) for (c, p) in m]


def verify_day_directory(name, camera):
    if name[:8].isnumeric() and name[16:24] == camera:
        return True
    elif name[:8].isnumeric():
        print(
            "WARNING: for CAMERA %s day directory name %s does not follow \
                name-convention of date_starttime.camera_* and will be ignored"
            % (camera, name)
        )
        return False
    else:  # all other directories are ignored
        return False


def get_days_in_order(interval=None, is_back=None, camera=None):
    """
    @params
    interval tuple (int i,int j) i < j, to return only days from i to j
    camera: concider days of the cameras folder, default: first camera,
        that expects all cameras to have the same number of cameras.
    """
    if camera is None or is_back is None:
        raise ValueError("provid kwargs is_back and camera")
    dir_ = get_directory(is_back)
    days = [
        name[:15]
        for name in os.listdir(dir_ + "/" + camera)
        if verify_day_directory(name, camera)
    ]
    days_unique = sorted(list(set(days)))
    if len(days_unique) < len(days):
        print(
            "WARNING DUPLICATE DAY: CAMERA %s_%s some days are duplicated, \
                please check the directory"
            % (camera, config.BACK if is_back else config.FRONT)
        )
    if interval:
        return days_unique[interval[0]: interval[1]]
    return days_unique


def read_batch_csv(filename, drop_errors):
    df = pd.read_csv(
        filename,
        skiprows=3,
        delimiter=";",
        usecols=["x", "y", "FRAME", "time", "xpx", "ypx"],
        dtype={"xpx": np.float64, "ypx": np.float64, "time": np.float64},
    )
    df.dropna(axis="rows", how="any", inplace=True)
    if drop_errors:
        err_filter = get_error_indices(df[:-1])
        df = df.drop(index=df[:-1][err_filter].index)
    df.reset_index(drop=True, inplace=True)
    return df


def get_center_radius_for_pods(
    fk: str, block: str, pod_locations_df: pd.DataFrame
) -> list:
    """
    Given a definition of a pods-location environment-variable, which specifies
    the path of a csv-file, and the current fishkey, this method resolves the
    center and the radius of the pod. Per individual in
    the PE-Experiment, 3 pods are used: a small, medium and large one.
    This method returns list of dict-elements for every pod.

    @params
    fk: str, ex. '23442333_front'
    block: str, ex. 'block1'
    pod_locations_df: pd.Dataframe
    @Returns
    [
        {"center": [1800, 750], "radius": 200},
        {"center": [1250, 1050], "radius": 25},
        {"center": [1000, 500], "radius": 15}
    ]
    """
    circular_walls = []
    pod_locs_list = get_pod_locations(
        fk, block, pod_locations_df
    )
    for pod_locs in pod_locs_list:
        point1 = int(pod_locs['origin_x']), int(pod_locs['origin_y'])
        point2 = int(pod_locs['end_x']), int(pod_locs['end_y'])
        center, radius = calculate_center_and_radius(point1, point2)
        circular_walls.append({
            "center": center, "radius": radius
        })
    return circular_walls


def get_pod_locations(
    fk: str, block: str, pod_locations_df: pd.DataFrame
) -> list:
    """
    Given a definition of a pods-location environment-variable, which specifies
    the path of a csv-file, and the current fishkey, this method resolves the
    origin and end x and y positions of the pod-locations. Per individual in the
    PE-Experiment, 3 pods are used: a small, medium and large one. 
    This method returns a dataframe for every position

    @params
    fk: str, ex. '23442333_front'
    block: str, ex. 'block1'
    pod_locations_df: pd.Dataframe
    @Returns
    [
        pd.DataFrame[s_xorigin, s_yorigin, s_xend, s_yend],
        pd.DataFrame[m_xorigin, m_yorigin, m_xend, m_yend]
        pd.DataFrame[l_xorigin, l_yorigin, l_xend, l_yend]
    ]
    """
    cam_id, compartment = fk.split('_')
    number_pattern = r'\d+'
    block_nr = re.search(number_pattern, block).group()
    
    rel_pod_locations = pod_locations_df[
        (
            pod_locations_df['block'] == int(block_nr)) & (
                pod_locations_df['camera_id'] == int(cam_id)) & (
                    pod_locations_df['front_or_back'].str.startswith(
                        compartment, na=False))
    ]
    small_loc_df = rel_pod_locations[rel_pod_locations['pot_size'] == 'small'][['origin_x', 'origin_y', 'end_x', 'end_y']]
    medium_loc_df = rel_pod_locations[rel_pod_locations['pot_size'] == 'medium'][['origin_x', 'origin_y', 'end_x', 'end_y']]
    large_loc_df = rel_pod_locations[rel_pod_locations['pot_size'] == 'large'][['origin_x', 'origin_y', 'end_x', 'end_y']]
    return small_loc_df, medium_loc_df, large_loc_df


def calculate_center_and_radius(point1, point2):
    # Normalization
    x1 = min(point1[0], point2[0])
    y1 = min(point1[1], point2[1])
    x2 = max(point1[0], point2[0])
    y2 = max(point1[1], point2[1])
    # center
    x_c = (x1 + x2) / 2
    y_c = (y1 + y2) / 2

    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # diagonal
    r = d / 2  # radius
    return (x_c, y_c), r


def get_error_indices(dataframe):
    """
    @params: dataframe
    returns a boolean pandas array with all indices to filter set to True
    """
    x = dataframe.xpx
    y = dataframe.ypx
    indexNames = ((x == -1) & (y == -1)) | (
        (x == 0) & (y == 0)
    )  # except the last index for time recording
    return indexNames


def merge_files(filenames, drop_errors):
    batches = []
    for f in filenames:
        df = read_batch_csv(f, drop_errors)
        batches.append(df)
    return batches


def csv_of_the_day(
    camera,
    day,
    is_back=False,
    drop_out_of_scope=False,
    batch_keys_remove=[],
    print_logs=False,
):
    """
    @params: camera, day, is_back, drop_out_of_scope
    returns csv of the day for camera: front or back
    """

    dir_ = get_directory(is_back=is_back)

    filenames_f = [
        f
        for f in glob.glob(
            "{}/{}/{}*/{}_{}*.csv".format(dir_, camera, day, camera, day),
            recursive=True,
        )
        if re.search(r"[0-9].*\.csv$", f[-6:])
    ]
    filenames_f = sorted(filenames_f)
    LOG, _, filtered_files = filter_files(
        camera,
        day,
        filenames_f,
        n_files=config.MAX_BATCH_IDX + 1,
        min_idx=config.MIN_BATCH_IDX,
    )  # filters for duplicates in the batches for a day & takes the LAST one
    for key in batch_keys_remove:
        filtered_files.pop(key, None)
    file_keys = list(filtered_files.keys())
    correct_files = list(filtered_files.values())
    if print_logs and len(LOG) > 0:
        print("\n {}/{}/{}*: \n".format(dir_, camera, day), "\n".join(LOG))
    return file_keys, merge_files(correct_files, drop_out_of_scope)


def filter_files(c, d, files, n_files=15, min_idx=0, Logger=None):
    """
    @params:
    c: camera_id
    d: folder name of a day
    files: list of files that are to be filtered
    n_files: number of files to expect.
    logger: Logger defined in path_validation
    @Returns: LOG, duplicate_f, correct_f
    msg_counter: number of debug-messages
    duplicate_f: a list of all duplicates occurring
    correct_f: dict of the correct files for keys i in 0,...,n_files-1
    """
    msg_counter = 0
    missing_numbers = []
    duplicate_f = []
    correct_f = dict()
    for i in range(min_idx, n_files):
        key_i = "{:06d}".format(i)
        pattern = re.compile(
            ".*{}_{}.{}_{}_\d*-\d*-\d*T\d*_\d*_\d*_\d*.csv".format(
                c, d[:15], c, key_i
            )
        )
        i_f = [f for f in files if pattern.match(f) is not None]

        if len(i_f) > 1:
            i_f.sort()
            duplicate_f.extend(i_f[:-1])
            correct_f[key_i] = i_f[-1]
        elif len(i_f) == 0:
            missing_numbers.append(key_i)
        else:
            correct_f[key_i] = i_f[-1]

    pattern_general = re.compile(
        ".*{}_{}.{}_\d*_\d*-\d*-\d*T\d*_\d*_\d*_\d*.csv".format(c, d[:15], c)
    )
    corrupted_f = [f for f in files if pattern_general.match(f) is None]

    if Logger and len(missing_numbers) > 0:
        msg_counter += 1
        Logger.debug(
            "The following files are missing: \n \t\t\t\t{}".format(
                " ".join(missing_numbers)
            )
        )
    if Logger and len(duplicate_f) > 0:
        msg_counter += 1
        Logger.debug(
            "The following files are duplicates: \n\t\t\t\t{}".format(
                "\n\t".join(duplicate_f)
            )
        )
    if Logger and len(corrupted_f) > 0:
        msg_counter += 1
        Logger.debug(
            "The following file names are corrupted, \
                 maybe wrong folder: \n\t\t\t\t{}".format(
                "\n\t".join(corrupted_f)
            )
        )
    return msg_counter, duplicate_f, correct_f


def start_time_of_day_to_seconds(START_TIME):
    """
    @start_time hhmmss
    return seconds (int)
    """
    if len(START_TIME) == 6:
        return (
            int(START_TIME[:2]) * 3600 + int(START_TIME[2:4]) * 60 + int(
                START_TIME[4:]
            )
        )
    else:
        raise ValueError("START_TIME must be of length 6")


def get_individuals_keys(parameters, block=""):
    files = glob.glob(
        parameters.projectPath+f"/Projections/{block}*_pcaModes.mat"
    )
    return sorted(list(set(map(
        lambda f: "_".join(f.split("/")[-1].split("_")[:3]),
        files
    ))))


def get_days(parameters, prefix=""):
    files = glob.glob(
        parameters.projectPath+f"/Projections/{prefix}*_pcaModes.mat"
    )
    return sorted(list(set(map(
        lambda f: "_".join(f.split("/")[-1].split("_")[3:5]),
        files
    ))))
