#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import cython
# tag: numpy
# You can ignore the previous line.
# It's for internal testing of the cython documentation.
from libc.math cimport acos, sqrt, ceil

import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t
ctypedef np.float64_t double
cdef int NDIM = 3

cdef double norm(double v0, double v1):
    return sqrt(v0**2 + v1**2)

cpdef np.ndarray[double, ndim=1] calc_steps(np.ndarray[double, ndim=2] data):
    sq = (data[1:] - data[:-1])**2
    c=np.sqrt(sq[:,0] + sq[:,1])
    return c

cpdef np.ndarray[double, ndim=1] tortuosity_of_chunk(np.ndarray[double, ndim=2] data):
    cdef int dist_length = 10 # normed by 10cm of distance traveled
    cdef np.ndarray[double, ndim=1] steps = calc_steps(data)
    cdef np.ndarray[double, ndim=1] c_steps = np.cumsum(steps)  # cumulative sum
    cdef list t_result = []
    cdef double L, C, curr_c
    cdef int i = 0
    cdef int j = 0
    cdef double min_L = 0.1
    curr_c = 0 # start with 0
    while i < c_steps.size-2:
        while j < c_steps.size-1 and c_steps[j]-curr_c < dist_length:
            j+=1
        L = np.sqrt(sum((data[j+1]-data[i])**2))
        C = c_steps[j] - curr_c
        if L < min_L: L=min_L
        if C < min_L: C=min_L
        if L > C: L=C
        t_result.append(C/L)
        curr_c = c_steps[j]
        i=j+1
        j=i

    return np.array(t_result, dtype=float)

cpdef (double, double) mean_std(np.ndarray[double, ndim=1] data):
    if data.size == 0:
        return (np.nan, np.nan)
    cdef double mean, std
    mean = data.sum()/data.size
    std = sqrt(((data-mean)**2).sum()/data.size)
    return (mean, std)

#### DISTANCE TO THE WALL --------------

cdef np.ndarray[double, ndim=2] distance_to_line(
    np.ndarray[double, ndim=1] x,
    np.ndarray[double, ndim=1] y,
    np.ndarray[double, ndim=1] a,
    np.ndarray[double, ndim=1] b, np.ndarray[double, ndim=1] c,
    np.ndarray[double, ndim=1] n):
    
    cdef int size = x.shape[0]
    cdef int n_lines = a.shape[0]
    cdef np.ndarray[double, ndim=2] dists = np.zeros((n_lines, size))
    cdef int i, j
    for i in range(n_lines):
        for j in range(size):
            dists[i, j] = abs(a[i] * x[j] + b[i] * y[j] + c[i]) / n[i]
    return dists

cdef double distance_to_circle(double[2] point, double[2] center, double radius):
    cdef double dx = point[0] - center[0]
    cdef double dy = point[1] - center[1]
    cdef double distance_to_center = sqrt(dx**2 + dy**2)
    cdef double distance_to_circle = max(0, distance_to_center - radius)
    return distance_to_circle

cdef np.ndarray[double, ndim=2] calc_wall_lines(np.ndarray[double, ndim=2] area):
    cdef np.ndarray[double, ndim=2] abcn = np.zeros((area.shape[0], 4))
    cdef int i
    cdef int size = area.shape[0]
    cdef double v1, v2, x, y
    for i in range(size):
        v1, v2 = area[(i+1) % size] - area[i]
        abcn[i, 0] = v2  # a
        abcn[i, 1] = -v1  # b
        x, y = area[i]
        abcn[i, 2] = y * v1 - v2 * x  # c
        abcn[i, 3] = norm(v2, v1)  # norm(a, b)
    return abcn

cdef np.ndarray[double, ndim=1] min_distance(np.ndarray[double, ndim=2] data, np.ndarray[double, ndim=2] abcn, list circular_walls):
    cdef np.ndarray[double, ndim=2] min_dists
    cdef np.ndarray[double, ndim=1] dists_to_lines
    cdef int i, j
    cdef double cx, cy, radius, dist_to_circle
    cdef double min_dist
    cdef double[2] point, center

    min_dists = distance_to_line(data[:, 0], data[:, 1], abcn[:, 0], abcn[:, 1], abcn[:, 2], abcn[:, 3])
    
    # Initialize the distances array with the minimum distance to lines
    dists_to_lines = np.min(min_dists, axis=0)

    # Calculate the distance to each circular wall and update the minimum distance
    for i in range(data.shape[0]):
        min_dist = dists_to_lines[i]
        point[0], point[1] = data[i, 0], data[i, 1]
        for circle in circular_walls:
            cx, cy = circle['center']
            radius = circle['radius']
            center[0], center[1] = cx, cy
            dist_to_circle = distance_to_circle(point, center, radius)
            if dist_to_circle < min_dist:
                min_dist = dist_to_circle
        dists_to_lines[i] = min_dist

    return dists_to_lines

cpdef np.ndarray[double, ndim=1] distance_to_wall_chunk(np.ndarray[double, ndim=2] data, np.ndarray[double, ndim=2] area, list circular_walls):
    cdef int size = data.shape[0]
    cdef np.ndarray[double, ndim=1] dists
    cdef np.ndarray[double, ndim=2] abcn = calc_wall_lines(area)
    dists = min_distance(data, abcn, circular_walls)
    return dists