import os
import shutil
import h5py
import hdf5storage
import numpy as np
import glob
from tqdm import tqdm

from utils import utils
import motionmapperpy as mmpy

def load_trajectory_data(parameters, fk="", day=""):
    data_by_day = []
    pfile = glob.glob(
        parameters.projectPath+f'/Projections/{fk}*_{day}*_pcaModes.mat'
    )
    pfile.sort()
    for f in tqdm(pfile):
        data = hdf5storage.loadmat(f)
        data_by_day.append(data)
    return data_by_day


def initialize_training_parameters():
    """
    Initializes the training parameters for the fish clustering toolbox.

    Returns:
        parameters (object): An object containing the initialized
        training parameters.

    Hints:
        trainingSetSize:
            Total number of training set points to find.
            Increase or decrease based on available RAM. For reference,
            36k is a good number with 64GB RAM.
        embedding_batchSize:
            Lower this if you get a memory error when
            re-embedding points on a learned map.
    """
    parameters = utils.set_parameters()
    parameters.training_numPoints = 5000  # Number of points in mini-trainings.
    parameters.trainingSetSize = 72000
    parameters.embedding_batchSize = 30000

    utils.createProjectDirectory(parameters.projectPath)

    return parameters


def return_normalization_func(parameters):
    """
    Returns a normalization function that normalizes projections
    based on the standard deviation of the data.

    Parameters:
        parameters (dict): A dictionary containing the parameters
        for loading trajectory data.

    Returns:
        function: A lambda function that takes a projection and
        returns the normalized projection.
    """
    data = np.concatenate(
        [d["projections"] for d in load_trajectory_data(parameters)]
    )
    std = data.std(axis=0)
    return lambda pro: pro/std


def subsample_from_projections(parameters):
    """
    Subsamples training data from projections and saves it in a directory.

    Args:
        parameters (object): An object containing the necessary parameters.

    Returns:
        tuple: A tuple containing the subsampled
        training data and training amplitudes.
    """
    projection_directory = parameters.projectPath+'/Projections/'
    tsne_directory = parameters.projectPath + '/UMAP/'

    if not os.path.exists(tsne_directory+'training_data.mat'):
        trainingSetData, trainingSetAmps, _ = mmpy.runEmbeddingSubSampling(
            projection_directory,
            parameters
        )
        if os.path.exists(tsne_directory):
            shutil.rmtree(tsne_directory)
            os.mkdir(tsne_directory)
        else:
            os.mkdir(tsne_directory)

        hdf5storage.write(
            data={'trainingSetData': trainingSetData},
            path='/',
            truncate_existing=True,
            filename=tsne_directory+'/training_data.mat',
            store_python_metadata=False,
            matlab_compatible=True
        )

        hdf5storage.write(
            data={'trainingSetAmps': trainingSetAmps},
            path='/',
            truncate_existing=True,
            filename=tsne_directory + '/training_amps.mat',
            store_python_metadata=False,
            matlab_compatible=True
        )
    else:
        # Subsampled trainingSetData found skipping minitSNE and running training tSNE
        with h5py.File(tsne_directory + '/training_data.mat', 'r') as hfile:
            trainingSetData = hfile['trainingSetData'][:].T
    return trainingSetData
