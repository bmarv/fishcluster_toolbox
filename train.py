import argparse
import hdf5storage
import glob
import time
import os
import motionmapperpy as mmpy
from training import training_processing
from training import embedding
from training import inferencing
from tqdm import tqdm
import wandb
import multiprocessing as mp
from multiprocessing import Pool


def run_training(n_neighbors, min_dist, threads_cpu, data=None):
    # PRE-PROCESSING
    parameters = training_processing.initialize_training_parameters()
    parameters.n_neighbors = n_neighbors
    parameters.min_dist = min_dist
    if data is not None:
        parameters.projectPath = data
    init_wandb(parameters)
    print("Data Normalization")
    parameters.normalize_func = training_processing\
        .return_normalization_func(parameters)
    print("Subsampling from Projections")
    trainingSetData = training_processing.subsample_from_projections(
        parameters
    )

    # EMBEDDING
    print('Embedding using UMAP and K-Means')
    trainingEmbedding, umap_model = embedding.run_UMAP(
        trainingSetData,
        parameters
    )
    trainingSetData[trainingSetData == 0] = 1e-12  # replace 0 with 1e-12
    models_directory = parameters.projectPath + '/Models/'
    kmeans_models = []
    for k in parameters.kmeans_list:
        kmeans_models.append(
            embedding.run_kmeans(k, models_directory, trainingSetData)
        )

    # INFERENCING
    print('Inferencing using UMAP')
    projectionFiles = glob.glob(
        parameters.projectPath + '/Projections/*pcaModes.mat'
    )
    if threads_cpu == -1:
        threads_cpu = mp.cpu_count() - 1  # all cores, if not specified
    elif threads_cpu == 0:
        threads_cpu = 1
    print(f'using {threads_cpu} cpus')

    inferencing_batches = [
        projectionFiles[i::threads_cpu] for i in range(threads_cpu)
    ]
    parameters.normalize_func = None
    with Pool(threads_cpu) as pool:
        _ = pool.starmap(
            inferencing_umap_f_batch,
            [(
                batch, parameters, umap_model
            ) for batch in inferencing_batches]
        )
        pool.close()
        pool.join()

    print('Inferencing using KMeans')
    for proj_file in tqdm(projectionFiles):
        if os.path.exists(proj_file[:-4] + f'_clusters_{parameters.kmeans_list[-1]}.mat'):
            continue

        projections = hdf5storage.loadmat(proj_file)['projections']
        inferencing.kmeans_inference_for_individual(
            projections,
            parameters,
            proj_file,
            kmeans_models
        )

    # SEGMENTATION
    print('Watershed Segmentation for UMAP')
    startsigma = 1.0
    for k in parameters.kmeans_list:
        print(f'Cluster-Size: {k}')
        mmpy.findWatershedRegions(
            parameters,
            minimum_regions=k,
            startsigma=startsigma,
            pThreshold=[0.33, 0.67],
            saveplot=True,
            endident='*_pcaModes.mat'
        )
    wandb.finish()


def init_wandb(params):
    wandb.init(
        project="pe_training",
        config=params
    )


def inferencing_umap_f_batch(
    inferencing_batch,
    parameters,
    umap_model
):
    current = mp.current_process()

    for proj_file in tqdm(
        inferencing_batch,
        desc=f'id: {os.getpid()}',
        position=current._identity[0]+1
    ):
        if os.path.exists(proj_file[:-4] + '_uVals.mat'):
            continue

        projections = hdf5storage.loadmat(proj_file)['projections']
        
        inferencing.umap_inference_for_individual(
            projections,
            parameters,
            proj_file,
            umap_model
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="embedding and inferencing using umap and kmeans"
    )
    parser.add_argument("--n_neighbors", required=True, type=int, default=15)
    parser.add_argument("--min_dist", required=True, type=float, default=0.1)
    parser.add_argument("--threads_cpu", required=False, type=int, default=-1)
    parser.add_argument("--data", required=False, type=str, default=None)
    args = parser.parse_args()

    n_neighbors = args.n_neighbors
    min_dist = args.min_dist
    threads_cpu = args.threads_cpu
    data = args.data

    print(f'n_neighbors: {n_neighbors}, min_dist: {min_dist}, threads_cpu: {threads_cpu},\
           data: {data}')
    run_training(n_neighbors, min_dist, threads_cpu, data)
