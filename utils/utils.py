import glob
import math
import os
import numpy as np
import bisect
from easydict import EasyDict as edict
from umap import UMAP
from config import projectPath


def pointsInCircum(r, n=100):
    return [(
        math.cos(2*math.pi/n*x)*r, math.sin(2*math.pi/n*x)*r
        ) for x in range(0, n+1)
    ]


def runs_of_ones_array(bits):
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts, run_starts, run_ends


def combine_ones_and_zeros(ones, zeros, th, size):
    block = 0
    hits = 0
    records = list()
    j, i = 0, 0
    while i < ones.shape[0]:
        if block >= size:
            if (hits/block) >= th:
                records.append({'idx': (j, i), 'score': (hits/block)})
                block, hits, j = 0, 0, i
            else:
                block -= (ones[j] + zeros[j])
                hits -= ones[j]
                j += 1
        if block == 0:
            block, hits = ones[i], ones[i]
            i += 1
        elif block < size:
            block += (ones[i] + zeros[i-1])
            hits += ones[i]
            i += 1
    return records


def get_cluster_sequences(
    clusters,
    cluster_ids=range(1, 6),
    sw=6*60*5,
    th=0.6
):
    records = dict(zip(cluster_ids, [list() for i in range(len(cluster_ids))]))
    for cid in cluster_ids:
        bits = clusters == cid
        n_ones, rs, re = runs_of_ones_array(bits)
        n_zeros = rs[1:] - re[:-1]
        matches = combine_ones_and_zeros(n_ones, n_zeros, th, sw)
        matches.sort(key=lambda x: x["score"], reverse=True)
        results = [(
            rs[m['idx'][0]], re[m['idx'][1]], m['score']) for m in matches
        ]
        records[cid].extend(results)
    return records


def split_into_batches(time, data, batch_size=60*60*5, flt=True):
    """
    @param time: time vector in data frames 

    Split data into batches of batch_size dataframes (5 df per second)
    flt: if True, filter out batches with less than 1/4 of the batch_size
    """
    t = time-time[0]
    step = int(batch_size)  # 5 df per second
    hours_end = [bisect.bisect_left(t, h) for h in range(
        step,
        int(t[-1]),
        step
    )]
    batches = np.split(data, hours_end)
    times = np.split(time, hours_end)
    batches = list(filter(lambda v: len(v) > 0, batches))
    times = list(filter(lambda v: len(v) > 0, times))
    return times, batches


def get_individuals_keys(parameters, block=""):
    files = glob.glob(
        parameters.projectPath
        + f"/Projections/{block}*_pcaModes.mat"
    )
    return sorted(list(set(map(
        lambda f: "_".join(f.split("/")[-1].split("_")[:3]), files
    ))))


def get_days(parameters, prefix=""):
    files = glob.glob(
        parameters.projectPath
        + f"/Projections/{prefix}*_pcaModes.mat")
    return sorted(list(set(map(
        lambda f: "_".join(f.split("/")[-1].split("_")[3:5]), files
    ))))


# inspired by bermans motionmapperpy
def setRunParameters(parameters=None):
    """
    Get parameter dictionary for running motionmapperpy.
    :param parameters: Existing parameter dictionary,
        defaults will be filled for missing keys.
    :return: Parameter dictionary.
    """
    if isinstance(parameters, dict):
        parameters = edict(parameters)
    else:
        parameters = edict()

    """# %%%%%%%% General Parameters %%%%%%%%"""

    # %number of processors to use in parallel code
    numProcessors = 10

    useGPU = -1

    method = 'UMAP'  # or 'UMAP'

    """%%%%%%%% Wavelet Parameters %%%%%%%%"""
    # %Whether to do wavelet decomposition,
    # if False then use normalized projections for tSNE embedding.
    waveletDecomp = True

    # %number of wavelet frequencies to use
    numPeriods = 25

    # dimensionless Morlet wavelet parameter
    omega0 = 5

    # sampling frequency (Hz)
    samplingFreq = 100

    # minimum frequency for wavelet transform (Hz)
    minF = 1

    # maximum frequency for wavelet transform (Hz)
    maxF = 50

    """%%%%%%%% t-SNE Parameters %%%%%%%%"""
    # Global tSNE method - 'barnes_hut' or 'exact'
    tSNE_method = 'barnes_hut'

    # %2^H (H is the transition entropy)
    perplexity = 32

    # %embedding batchsize
    embedding_batchSize = 20000

    # %maximum number of iterations for the Nelder-Mead algorithm
    maxOptimIter = 100

    # %number of points in the training set
    trainingSetSize = 35000

    # %number of neigbors to use when re-embedding
    maxNeighbors = 200

    # %local neighborhood definition in training set creation
    kdNeighbors = 5

    # %t-SNE training set perplexity
    training_perplexity = 20

    # %number of points to evaluate in each training set file
    training_numPoints = 10000

    # %minimum training set template length
    minTemplateLength = 1

    """%%%%%%%% UMAP Parameters %%%%%%%%"""
    # Size of local neighborhood for UMAP.
    n_neighbors = 15

    # Negative sample rate while training.
    train_negative_sample_rate = 5

    # Negative sample rate while embedding new data.
    embed_negative_sample_rate = 1

    # Minimum distance between neighbors.
    min_dist = 0.1

    # UMAP output dimensions.
    umap_output_dims = 2

    # Number of training epochs.
    n_training_epochs = 1000

    # Embedding rescaling parameter.
    rescale_max = 100
    if "kmeans_list" not in parameters:
        parameters.kmeans_list = [10]
    if "umap_method" not in parameters:
        parameters.umap_module = UMAP

    if "normalize_func" not in parameters.keys():
        parameters.normalize_func = None

    if "kmeans" not in parameters.keys():
        parameters.n_clusters = None

    if "numProcessors" not in parameters.keys():
        parameters.numProcessors = numProcessors

    if "numPeriods" not in parameters.keys():
        parameters.numPeriods = numPeriods

    if "omega0" not in parameters.keys():
        parameters.omega0 = omega0

    if "samplingFreq" not in parameters.keys():
        parameters.samplingFreq = samplingFreq

    if "minF" not in parameters.keys():
        parameters.minF = minF

    if "maxF" not in parameters.keys():
        parameters.maxF = maxF

    if "tSNE_method" not in parameters.keys():
        parameters.tSNE_method = tSNE_method

    if "perplexity" not in parameters.keys():
        parameters.perplexity = perplexity

    if "embedding_batchSize" not in parameters.keys():
        parameters.embedding_batchSize = embedding_batchSize

    if "maxOptimIter" not in parameters.keys():
        parameters.maxOptimIter = maxOptimIter

    if "trainingSetSize" not in parameters.keys():
        parameters.trainingSetSize = trainingSetSize

    if "maxNeighbors" not in parameters.keys():
        parameters.maxNeighbors = maxNeighbors

    if "kdNeighbors" not in parameters.keys():
        parameters.kdNeighbors = kdNeighbors

    if "training_perplexity" not in parameters.keys():
        parameters.training_perplexity = training_perplexity

    if "training_numPoints" not in parameters.keys():
        parameters.training_numPoints = training_numPoints

    if "minTemplateLength" not in parameters.keys():
        parameters.minTemplateLength = minTemplateLength

    if "waveletDecomp" not in parameters.keys():
        parameters.waveletDecomp = waveletDecomp

    if "useGPU" not in parameters.keys():
        parameters.useGPU = useGPU

    if "n_neighbors" not in parameters.keys():
        parameters.n_neighbors = n_neighbors

    if "train_negative_sample_rate" not in parameters.keys():
        parameters.train_negative_sample_rate = train_negative_sample_rate

    if 'embed_negative_sample_rate' not in parameters.keys():
        parameters.embed_negative_sample_rate = embed_negative_sample_rate

    if 'min_dist' not in parameters.keys():
        parameters.min_dist = min_dist

    if 'umap_output_dims' not in parameters.keys():
        parameters.umap_output_dims = umap_output_dims

    if 'n_training_epochs' not in parameters.keys():
        parameters.n_training_epochs = n_training_epochs

    if 'rescale_max' not in parameters.keys():
        parameters.rescale_max = rescale_max

    if 'method' not in parameters.keys():
        parameters.method = method

    return parameters


def createProjectDirectory(pathToProject):
    _dirs = [
        pathToProject,
        f'{pathToProject}/Projections',
        f'{pathToProject}/UMAP'
    ]
    for d in _dirs:
        if not os.path.exists(d):
            print(f'Creating : {d}')
            os.mkdir(d)
        else:
            print(f'Skipping, path already exists : {d}')
    return


def set_parameters(parameters=None): 
    parameters = setRunParameters(parameters)
    parameters.pcaModes = 3
    parameters.samplingFreq = 5
    parameters.maxF = 2.5
    parameters.minF = 0.01
    parameters.omega0 = 5
    parameters.numProcessors = 10
    parameters.method = "UMAP"
    parameters.kmeans = 10
    parameters.kmeans_list = [5, 7, 10, 20, 50, 100]
    parameters.projectPath = projectPath
    os.makedirs(parameters.projectPath, exist_ok=True)
    createProjectDirectory(parameters.projectPath)
    return parameters
