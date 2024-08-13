import numpy as np
import pickle
import warnings


def run_UMAP(data, parameters, save_model=True):
    """
    Runs UMAP (Uniform Manifold Approximation and Projection)
    on the given data.
    Based on motionmapperpy.

    Args:
        data (numpy.ndarray): The input data to be embedded using UMAP.
        parameters (dict): A dictionary containing the UMAP parameters.
        save_model (bool, optional): Whether to save the UMAP model to disk

    Returns:
        numpy.ndarray: The embedded data after applying UMAP.

    Raises:
        ValueError: If UMAP is not implemented without wavelet decomposition.
    """

    if not parameters.waveletDecomp:
        raise ValueError('UMAP not implemented without wavelet decomposition.')

    vals = np.sum(data, 1)
    if ~np.all(vals == 1):
        data = data / vals[:, None]

    UMAP = parameters.umap_module

    um = UMAP(
        n_neighbors=parameters['n_neighbors'],
        negative_sample_rate=parameters['train_negative_sample_rate'],
        min_dist=parameters['min_dist'],
        n_components=parameters['umap_output_dims'],
        n_epochs=parameters['n_training_epochs']
    )
    y = um.fit_transform(data)
    trainmean = np.mean(y, 0)
    scale = (parameters['rescale_max'] / np.abs(y).max())
    y = y - trainmean
    y = y * scale

    if save_model:
        modelsfolder = parameters['projectPath'] + '/Models/'
        np.save(
            modelsfolder + '_trainMeanScale.npy',
            np.array([trainmean, scale], dtype=object)
        )
        with open(modelsfolder + 'umap.model', 'wb') as f:
            pickle.dump(um, f)

    return y, um


def run_kmeans(parameters, k, models_directory, trainingSetData):
    """
    Runs the K-means clustering algorithm on the training set data.
    Based on motionmapperpy.

    Args:
        k (int): The number of clusters to create.
        tsne_directory (str): The directory to save the K-means model.
        trainingSetData (array-like): The training set data to be clustered.

    Returns:
        kmeans: The trained K-means model.
    """
    if parameters.useGPU >= 0:
        try:
            from cuml import KMeans
        except ModuleNotFoundError as E:
            warnings.warn("Trying to use GPU but cuml is not installed.\
                Install cuml or set parameters.useGPU = -1. ")
            raise E
    else:
        from sklearn.cluster import MiniBatchKMeans as KMeans

    kmeans = KMeans(
        n_clusters=k,
        random_state=0
    ).fit(trainingSetData)
    pickle.dump(
        kmeans,
        open(models_directory + f"/kmeans_{k}.pkl", "wb")
    )
    return kmeans
