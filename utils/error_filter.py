import numpy as np
import config_processing as config
from utils.processing_methods import distance_to_wall_chunk, calc_steps
from fishproviz.utils.transformation import px2cm


def all_error_filters(data, area_tuple, **kwargs):
    """
    Summery of all error filters
    @params: dataframe
    returns a boolean numpy array with all indices to filter out -- set to True!
    """
    data_flt = error_default_points(data)
    if config.AREA_FILTER:
        data_flt = data_flt | error_points_out_of_area(data, area_tuple)
    if config.DIRT_FILTER:
        data_flt = data_flt | error_dirt_points(data, **kwargs)
    return data_flt


def error_default_points(data):
    x, y = data[:, 0], data[:, 1]
    # filter nan
    nan_filter = np.isnan(x) | np.isnan(y)
    return ((x == -1) & (y == -1)) | ((x == 0) & (y == 0)) | nan_filter


def error_dirt_points(data, threshold=config.DIRT_THRESHOLD, fish_key="", day=""):  #
    """
    @params:    data -- numpy array with x,y coordinates
                threshold -- number of data frames that are sequentially equal such that we classify them as dirt.
    returns a boolean pandas array with all indices to filter set to True
    """
    bool_array = np.all(data[1:] == data[:-1], axis=1)
    flt = np.zeros(data.shape[0], dtype=bool)
    indexer = np.argwhere(np.diff(bool_array)).squeeze()
    start = 0
    for end in [*(indexer + 2), data.shape[0]]:
        if (
            bool_array[start] and (end - start) > threshold
        ):  # if the first element is true (hence it is a beginning of an equal sequence) and the difference is greater than the threshold
            flt[start:end] = True
            x, y = data[start]
            if ~(((x == -1) & (y == -1)) | ((x == 0) & (y == 0))):
                msg_spike_s, msg_spike_e = "", ""
                if start > 0:
                    spike_s = calc_steps(data[start - 1 : start + 1])[0]
                    if px2cm(spike_s) > config.SPIKE_THRESHOLD:
                        msg_spike_s = "SPIKE START"
                if end < data.shape[0]:
                    spike_e = calc_steps(data[end - 1 : end + 1])[0]
                    if px2cm(spike_e) > config.SPIKE_THRESHOLD:
                        msg_spike_e = "SPIKE END"
                print(
                    "DIRT: %s, %s: Found dirt: %d data points for [%d:%d] out of %d, data[start]=[%d,%d] \t %s \t %s"
                    % (
                        fish_key,
                        day,
                        end - start,
                        start,
                        end,
                        data.shape[0],
                        *data[start],
                        msg_spike_s,
                        msg_spike_e,
                    ),
                    "%.2f" % px2cm(spike_s),
                    "%.2f" % px2cm(spike_e),
                )
                print(
                    "dirt threshold: %.02f" % (threshold / (5 * 60)),
                    "min",
                    "Dirt Sequence %.02f" % ((end - start) / (5 * 60)),
                    "min",
                )
                with open(config.err_file, "a") as f:
                    f.write(
                        ";".join(
                            [
                                fish_key,
                                day,
                                str((end - start) / (5 * 60)),
                                str(data[start][0]),
                                str(data[start][1]),
                                str(start),
                                str(end),
                            ]
                        )
                        + "\n"
                    )
        start = end - 1
    return flt


def error_points_out_of_range(data, area_tuple):
    # error out of range
    area = area_tuple[1]
    th = config.THRESHOLD_AREA_PX
    xmin, xmax = min(area[:, 0]) - th, max(area[:, 0]) + th
    ymin, ymax = min(area[:, 1]) - th, max(area[:, 1]) + th
    error_out_of_range = (
        (data[:, 0] < xmin)
        | (data[:, 0] > xmax)
        | (data[:, 1] < ymin)
        | (data[:, 1] > ymax)
    )
    return error_out_of_range


def error_points_out_of_area(data, area_tuple, day=""):
    """returns a boolean np.array, where true indecates weather the corresponding datapoint is on the wrong side of the tank"""
    key, area = area_tuple
    is_back = config.BACK in key  # key in the shape of <<camera>>_<<position>>
    error_out_of_range = error_points_out_of_range(data, area_tuple)
    # over the diagonal
    AB = area[2] - area[1]
    AP = data - area[1]
    if is_back:
        error_filter = (
            AP[:, 1] * AB[0] - AP[:, 0] * AB[1] <= 0
        )  # cross product (a1b2−a2b1)
    else:
        error_filter = (
            AP[:, 1] * AB[0] - AP[:, 0] * AB[1] >= 0
        )  # cross product (a1b2−a2b1)
    error_filter = error_filter

    err_default = error_default_points(data)
    error_non_default = error_filter & ~err_default
    error_non_default[error_non_default] = (
        distance_to_wall_chunk(data[error_non_default], area) > config.THRESHOLD_AREA_PX
    )  # in pixels
    error_non_default = error_non_default | (error_out_of_range & ~err_default)
    if np.any(error_non_default):  # ef and not ed
        print(
            "AREA: %s, %s %d dataframes out of %d where on the other side of the tank. They are beeing filtered out."
            % (key, day, error_non_default.sum(), data.shape[0])
        )
    return error_non_default
