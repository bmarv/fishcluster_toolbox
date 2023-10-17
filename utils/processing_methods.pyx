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

#### DINSTANCE TO THE WALL --------------

cpdef np.ndarray[double, ndim=1] distance_to_wall_chunk(np.ndarray[double, ndim=2] data, np.ndarray[double, ndim=2] area):
    cdef int size
    size = data.shape[0]
    cdef np.ndarray[double, ndim=1] dists
    cdef np.ndarray[double, ndim=2] abcn = calc_wall_lines(area)
    #for i in range(size):
    dists = min_distance(data, abcn)
    return dists

cdef np.ndarray[double, ndim=2] calc_wall_lines(np.ndarray[double, ndim=2] area):
    cdef np.ndarray[double, ndim=2] abcn = np.zeros((area.shape[0], 4))
    cdef int i
    cdef int size = area.shape[0]
    cdef double v1, v2, x, y
    for i in range(size):
        v1,v2 = area[(i+1) % size]-area[i]
        abcn[i,0] = v2 # a
        abcn[i,1] = -v1 # b
        x,y = area[i]
        abcn[i,2] = y*v1-v2*x # c
        abcn[i,3]=norm(v2,v1) # norm(a,b)
    return abcn

cdef np.ndarray[double, ndim=1] min_distance(np.ndarray[double, ndim=2] data, np.ndarray[double, ndim=2] abcn):
    cdef np.ndarray[double, ndim=2] min_dists
    min_dists = distance_to_line(data[:,0], data[:,1], abcn[:,0], abcn[:,1], abcn[:,2], abcn[:,3])
    return np.min(min_dists, axis=0)

cdef np.ndarray[double, ndim=2] distance_to_line(
    np.ndarray[double, ndim=1] x,
    np.ndarray[double, ndim=1] y,
    np.ndarray[double, ndim=1] a,
    np.ndarray[double, ndim=1] b, np.ndarray[double, ndim=1] c,
    np.ndarray[double, ndim=1] n):
    return np.abs(a[:,np.newaxis]*x+b[:,np.newaxis]*y+c[:,np.newaxis])/n[:,np.newaxis]
