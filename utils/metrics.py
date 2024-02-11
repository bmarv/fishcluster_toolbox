import numpy as np
import config_processing as config


def get_spikes_filter(steps):
    return steps > config.SPIKE_THRESHOLD


def update_filter_two_points(steps, filter_index):
    return filter_index[:-1] | filter_index[1:] | get_spikes_filter(steps)


def update_filter_three_points(steps, filter_index):
    filter_index = update_filter_two_points(steps, filter_index)
    return filter_index[:-1] | filter_index[1:]


def compute_step_lengths(points):
    # Calculate the Euclidean distance between consecutive points
    vectors = np.diff(points, axis=0)
    distances = np.linalg.norm(vectors, axis=1)
    return distances


def calc_step_per_frame(batchxy, frames):
    """This function calculates the eucleadian step length in centimeters
    per FRAME, this is useful as a speed measurement after the removal
    of erroneous data points.
    """
    frame_dist = frames[1:] - frames[:-1]
    c = compute_step_lengths(batchxy) / frame_dist
    return c


def compute_turning_angles(points):
    # Compute the differences between adjacent points
    vectors = np.diff(points, axis=0)

    # Find the indices where the difference vector is non-zero and finit
    wanted_indices = np.any(vectors != 0, axis=1) \
        & np.all(np.isfinite(vectors), axis=1)

    vectors = vectors[wanted_indices]
    # Compute the dot products and determinants between pairs of vectors
    dot_products = np.einsum("ij,ij->i", vectors[:-1], vectors[1:])
    determinants = np.cross(vectors[:-1], vectors[1:])

    # Compute the turning angles
    turning_angles = np.arctan2(determinants, dot_products)
    turning_angles_result = np.zeros(points.shape[0] - 2)
    # the last one is buried in the angle if not False anyways
    wanted_angles = np.where(wanted_indices)[0][1:] - 1
    # Set the turning angles to 0 for equal consecutive points
    turning_angles_result[wanted_angles] = turning_angles
    return turning_angles_result


def entropy_heatmap(chunk, area, bins=(18, 18)):
    """Calculate the 2D histogram of the chunk"""
    th = config.THRESHOLD_AREA_PX
    xmin, xmax = min(area[:, 0]) - th, max(area[:, 0]) + th
    ymin, ymax = min(area[:, 1]) - th, max(area[:, 1]) + th

    return np.histogram2d(
        chunk[:, 0],
        chunk[:, 1],
        bins=bins,
        density=False,
        range=[[xmin, xmax], [ymin, ymax]],
    )[0]
