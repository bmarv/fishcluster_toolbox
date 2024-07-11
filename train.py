import gc
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
import torch


def run_training(n_neighbors, min_dist, data, device):
    # PRE-PROCESSING
    tall = time.time()
    parameters = training_processing.initialize_training_parameters()
    parameters.n_neighbors = n_neighbors
    parameters.min_dist = min_dist
    parameters.projectPath = data
    parameters.useGPU = device  # 0 for GPU, -1 for CPU
    init_wandb(parameters)
    if parameters.useGPU == 0:
        from cuml import UMAP  # GPU
    else:
        from umap import UMAP
    parameters.umap_module = UMAP
    mmpy.createProjectDirectory(parameters.projectPath)
    print("Data Normalization")
    parameters.normalize_func = training_processing\
        .return_normalization_func(parameters)
    print("Subsampling from Projections")
    trainingSetData = training_processing.subsample_from_projections(parameters)

    # EMBEDDING
    print('Embedding using UMAP and K-Means')
    trainingEmbedding = embedding.run_UMAP(trainingSetData, parameters)
    trainingSetData[trainingSetData == 0] = 1e-12  # replace 0 with 1e-12
    tfolder = parameters.projectPath + f'/{parameters.method}/'
    for k in parameters.kmeans_list:
        if not os.path.exists(tfolder + f'/kmeans_{k}.pkl'):
            embedding.run_kmeans(k, tfolder, trainingSetData, parameters.useGPU)

    # INFERENCING
    print('Inferencing using UMAP and K-Means')
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
            projectionFiles[i]
        )
        inferencing.umap_inference_for_individual(
            projections,
            trainingSetData,
            trainingEmbedding,
            parameters,
            projectionFiles[i]
        )
        wandb.log({"inferencing/step": i})

    print(f'Embedding and Inference finished in {time.time() - tall} seconds!')

    # SEGMENTATION
    print('Watershed Segmentation using UMAP')
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="embedding and inferencing using umap and kmeans"
    )
    parser.add_argument("--n_neighbors", required=True, type=int, default=15)
    parser.add_argument("--min_dist", required=True, type=float, default=0.1)
    parser.add_argument("--data", required=True, type=str, default='/mnt')
    parser.add_argument("--device", required=True, type=int, default=0)
    args = parser.parse_args()

    n_neighbors = args.n_neighbors
    min_dist = args.min_dist
    data = args.data
    device = args.device
    
    print(f'n_neighbors: {n_neighbors}, min_dist: {min_dist}, data: {data}, device: {device}')
    run_training(n_neighbors, min_dist, data, device)