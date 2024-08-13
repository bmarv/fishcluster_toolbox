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
import torch
import gc


def run_training(parameters):
    tall = time.time()
    # PRE-PROCESSING
    if parameters.wandb_key:
        init_wandb(parameters)
    parameters.useGPU = parameters.device
    if parameters.useGPU == 0:
        from cuml import UMAP
    else:
        from umap import UMAP
    parameters.umap_module = UMAP
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
            embedding.run_kmeans(
                parameters,
                k,
                models_directory,
                trainingSetData
            )
        )

    print("Inferencing using UMAP and KMeans")
    projectionFiles = glob.glob(
        parameters.projectPath + '/Projections/*pcaModes.mat'
    )
    for i in tqdm(range(len(projectionFiles)), total=len(projectionFiles)):
        if os.path.exists(projectionFiles[i][:-4] + '_uVals.mat'):
            continue

        # manual garbage collection for cuml
        gc.collect()
        torch.cuda.empty_cache()

        projections = hdf5storage.loadmat(projectionFiles[i])['projections']
        inferencing.kmeans_inference_for_individual(
            projections,
            parameters,
            projectionFiles[i],
            kmeans_models
        )
        inferencing.umap_inference_for_individual(
            projections,
            parameters,
            projectionFiles[i],
            umap_model
        )
        wandb.log({"inferencing/step": i})

    print(f'Embedding and Inference finished in {time.time() - tall} seconds!')

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
    if parameters.wandb_key:
        wandb.finish()


def init_wandb(params):
    wandb.init(
        project="fishcluster_toolbox",
        config=params
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="embedding and inferencing using umap and kmeans"
    )
    parser.add_argument("--n_neighbors", required=False, type=int)
    parser.add_argument("--min_dist", required=False, type=float)
    parser.add_argument("--threads_cpu", required=False, type=int)
    parser.add_argument("--data", required=False, type=str)
    parser.add_argument("--device", required=False, type=int)
    args = parser.parse_args()

    parameters = training_processing.initialize_training_parameters()
    if args.n_neighbors is not None:
        parameters.n_neighbors = args.n_neighbors
    if args.min_dist is not None:
        parameters.min_dist = args.min_dist
    if args.threads_cpu is not None:
        parameters.threads_cpu = args.threads_cpu
    if args.data is not None:
        parameters.projectPath = args.data
    if args.device is not None:
        parameters.device = args.device

    if parameters.threads_cpu == -1:
        parameters.threads_cpu = mp.cpu_count()  # all cores
    elif parameters.threads_cpu == 0:
        parameters.threads_cpu = 1

    print(f'''
        n_neighbors: {parameters.n_neighbors}
        min_dist: {parameters.min_dist}
        threads_cpu: {parameters.threads_cpu}
        data: {parameters.projectPath}
        device: {parameters.device}
    ''')
    run_training(parameters)
