import gc
import numpy as np
import pickle
import hdf5storage
from easydict import EasyDict as edict

import motionmapperpy as mmpy


def umap_inference_for_individual(
    projections,
    parameters,
    projectionFile,
    umap_model
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
    """
    numModes = parameters.pcaModes
    if parameters.waveletDecomp:
        # Finding Wavelets
        data, f = mmpy.motionmapper.mm_findWavelets(
            projections,
            numModes,
            parameters
        )
    data = data / np.sum(data, 1)[:, None]
    modelsfolder = parameters['projectPath'] + '/Models/'
    um = umap_model
    trainparams = np.load(
        modelsfolder + '_trainMeanScale.npy',
        allow_pickle=True
    )
    um.negative_sample_rate = parameters['embed_negative_sample_rate']
    zValues = um.transform(data)
    gc.collect()
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
    del zValues
    with open(
        projectionFile[:-4] + '_uVals_outputStatistics.pkl',
        'wb'
    ) as hfile:
        pickle.dump(outputStatistics, hfile)
    del outputStatistics


def kmeans_inference_for_individual(
    projections,
    parameters,
    projectionFile,
    kmeans_models
):
    """
    Perform k-means inference for individual projections.
    Based on motionmapperpy.

    Args:
        projections (numpy.ndarray): The input projections.
        parameters (object): The parameters object containing
            configuration settings.
        projectionFile (str): The file path of the projection file.

    """
    numModes = parameters.pcaModes
    if parameters.waveletDecomp:
        data, f = mmpy.motionmapper.mm_findWavelets(
            projections,
            numModes,
            parameters
        )
    clusters_dict = {}
    for idx, k in enumerate(parameters.kmeans_list):
        clusters_dict[f"clusters_{k}"] = kmeans_models[idx].predict(data)

    for key, value in clusters_dict.items():
        hdf5storage.write(
            data={"clusters": value, "k": int(key.split("_")[1])},
            path='/',
            truncate_existing=True,
            filename=projectionFile[:-4]+'_%s.mat' % (key),
            store_python_metadata=False,
            matlab_compatible=True
        )
    del clusters_dict
    gc.collect()
