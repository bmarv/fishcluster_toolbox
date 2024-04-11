import time
import numpy as np
import pickle
import hdf5storage
from easydict import EasyDict as edict

import motionmapperpy as mmpy


def umap_inference_for_individual(
    projections,
    trainingData,
    trainingEmbedding,
    parameters,
    projectionFile,
):
    """
    Perform umap inference for individual projections.
    Based on motionmapperpy.

    :param projections:  N x (pcaModes x numPeriods) array of
        projection values.
    :param trainingData: Nt x (pcaModes x numPeriods) array of wavelet
        amplitudes containing Nt data points.
    :param trainingEmbedding: Nt x 2 array of embeddings.
    :param parameters: motionmapperpy Parameters dictionary.
    :return: zValues : N x 2 array of embedding results, outputStatistics :
        dictionary containing other parametric
    outputs.
    """
    numModes = parameters.pcaModes

    if parameters.waveletDecomp:
        # Finding Wavelets
        data, f = mmpy.motionmapper.mm_findWavelets(
            projections,
            numModes,
            parameters
        )
        if parameters.useGPU >= 0:
            data = data.get()
    else:
        # Using projections for tSNE. No wavelet decomposition
        f = 0
        data = projections
    data = data / np.sum(data, 1)[:, None]

    umapfolder = parameters['projectPath'] + '/UMAP/'
    with open(umapfolder + 'umap.model', 'rb') as f:
        um = pickle.load(f)
    trainparams = np.load(
        umapfolder + '_trainMeanScale.npy',
        allow_pickle=True
    )
    embed_negative_sample_rate = parameters['embed_negative_sample_rate']
    um.negative_sample_rate = embed_negative_sample_rate
    zValues = um.transform(data)
    zValues = zValues - trainparams[0]
    zValues = zValues * trainparams[1]
    outputStatistics = edict()
    outputStatistics.training_mean = trainparams[0]
    outputStatistics.training_scale = trainparams[1]

    del data

    hdf5storage.write(
        data={'zValues': zValues},
        path='/',
        truncate_existing=True,
        filename=projectionFile[:-4]+'_uVals.mat',
        store_python_metadata=False,
        matlab_compatible=True
    )

    # Save output statistics
    with open(
        projectionFile[:-4] + '_uVals_outputStatistics.pkl',
        'wb'
    ) as hfile:
        pickle.dump(outputStatistics, hfile)
    return zValues, outputStatistics


def kmeans_inference_for_individual(projections, parameters, projectionFile):
    """
    Perform k-means inference for individual projections.
    Based on motionmapperpy.

    Args:
        projections (numpy.ndarray): The input projections.
        parameters (object): The parameters object containing
            configuration settings.
        projectionFile (str): The file path of the projection file.

    Returns:
        dict: A dictionary containing the clustering results for
            different values of k.
    """
    t1 = time.time()
    numModes = parameters.pcaModes
    if parameters.waveletDecomp:
        # Finding Wavelets
        data, f = mmpy.motionmapper.mm_findWavelets(
            projections,
            numModes,
            parameters
        )
        if parameters.useGPU >= 0:
            data = data.get()
    else:
        # Using projections for tSNE. No wavelet decomposition
        data = projections
        data = data / np.sum(data, 1)[:, None]

    def kmeans(k):
        return pickle.load(
            open(
                parameters.projectPath
                + "/"
                + parameters.method
                + f"/kmeans_{k}.pkl", "rb"
                )
            )

    clusters_dict = dict(
        [(
            f"clusters_{k}",
            kmeans(k).predict(data)
            ) for k in parameters.kmeans_list]
    )
    for key, value in clusters_dict.items():
        hdf5storage.write(
            data={"clusters": value, "k": int(key.split("_")[1])},
            path='/',
            truncate_existing=True,
            filename=projectionFile[:-4]+'_%s.mat' % (key),
            store_python_metadata=False,
            matlab_compatible=True
        )
    return clusters_dict
