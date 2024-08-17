import numpy as np
from utils.tank_area_config import get_area_functions, \
    get_calibration_functions
import config

FUNCS_PX2CM = None
AREA_FUNCS = None


def normalize_origin_of_compartment(data, area, circular_walls, is_back):
    if is_back:
        origin1 = area[0, 0], area[1, 1]
        new_area = area - origin1
        origin2 = new_area[2, 0], new_area[3, 1]
        new_area = -new_area + origin2
        data = -data + origin1 + origin2
        
        center_els = []
        for pod_el in circular_walls:
            center_els.append(
                - np.array(pod_el["center"]) + origin1 + origin2
            )
        np_centers = np.array(center_els)

        # Mirror
        # find mirror line
        original_data_shape = data.shape
        data = np.vstack((data, new_area, np_centers))  # append area to mirror it as well
        long_edge_midpoint = (new_area[1] + new_area[2]) / 2
        mirror_line_slope = (long_edge_midpoint[1] - new_area[-1, 1]) / (
            long_edge_midpoint[0] - new_area[-1, 0]
        )
        mirror_line_intercept = new_area[-1, 1] - (
            mirror_line_slope * new_area[-1, 0]
        )
        # find line perpendicular to mirror line for each point
        orthogonal_mirror_line_slope = -1 / mirror_line_slope
        orthogonal_mirror_line_intercept = (
            data[:, 1] - orthogonal_mirror_line_slope * data[:, 0]
        )
        # find intersection of 2 lines for each point
        intersections_x = (
            orthogonal_mirror_line_intercept - mirror_line_intercept) / (
            mirror_line_slope - orthogonal_mirror_line_slope
        )
        intersections_y = (
            mirror_line_slope * orthogonal_mirror_line_intercept
            - orthogonal_mirror_line_slope * mirror_line_intercept
        ) / (mirror_line_slope - orthogonal_mirror_line_slope)
        # find mirror image of each point equidistant from intersection
        mirrored_data_x = 2 * intersections_x - data[:, 0]
        mirrored_data_y = 2 * intersections_y - data[:, 1]
        mirrored_data = np.vstack((mirrored_data_x, mirrored_data_y)).T

        data = mirrored_data[:original_data_shape[0]]
        new_area = mirrored_data[original_data_shape[0]: -np_centers.shape[0]]
        new_center = mirrored_data[data.shape[0] + new_area.shape[0]:]

        # TODO: mirror pods-locations
        new_circular_walls = []
        for idx, pod_el in enumerate(circular_walls):
            new_circular_walls.append({
                "center": new_center[idx],
                "radius": pod_el["radius"]
            })

    else:
        origin1 = area[1, 0], area[0, 1]
        new_area = area - origin1
        data = data - origin1
       
        new_circular_walls = []
        for pod_el in circular_walls:
            new_center = np.array(pod_el["center"]) - origin1

            new_circular_walls.append({
                "center": new_center,
                "radius": pod_el["radius"]
            })
    return data, new_area, new_circular_walls


def rotation(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def px2cm(a, fish_key=None):
    global FUNCS_PX2CM
    if not FUNCS_PX2CM:
        FUNCS_PX2CM = get_calibration_functions()
    if fish_key:
        return a * FUNCS_PX2CM(fish_key)
    return a * config.DEFAULT_CALIBRATION


def pixel_to_cm(pixels, fish_key=None):
    """
    @params: pixels (Nx2)
    returns: cm (Nx2)
    """
    global FUNCS_PX2CM
    global AREA_FUNCS
    if not AREA_FUNCS:
        AREA_FUNCS = get_area_functions()
    if AREA_FUNCS(fish_key) is None:
        origin = np.array([450, 450])  # default origin
    else:
        origin = AREA_FUNCS(fish_key)[1]  # origin of area is the 2nd point
    pixels = pixels - origin
    if not FUNCS_PX2CM:
        FUNCS_PX2CM = get_calibration_functions()
    R = rotation(np.pi / 4)
    if fish_key:
        t = [FUNCS_PX2CM(fish_key), FUNCS_PX2CM(fish_key)]
    else:
        t = [config.DEFAULT_CALIBRATION, config.DEFAULT_CALIBRATION]
    T = np.diag(t)
    cm_data = pixels @ R @ T
    return cm_data
