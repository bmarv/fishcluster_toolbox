import hdf5storage
import glob
import time
import os
import motionmapperpy as mmpy
from training import training_processing
from training import embedding
from training import inferencing

# PRE-PROCESSING
tall = time.time()
parameters = training_processing.initialize_training_parameters()
parameters.useGPU = -1  # 0 for GPU, -1 for CPU
mmpy.createProjectDirectory(parameters.projectPath)
parameters.normalize_func = training_processing\
    .return_normalization_func(parameters)
print("Subsample from projections")
trainingSetData = training_processing.subsample_from_projections(parameters)


# EMBEDDING
trainingEmbedding = embedding.run_UMAP(trainingSetData, parameters)
trainingSetData[trainingSetData == 0] = 1e-12  # replace 0 with 1e-12
tfolder = parameters.projectPath + f'/{parameters.method}/'
for k in parameters.kmeans_list:
    if not os.path.exists(tfolder + f'/kmeans_{k}.pkl'):
        embedding.run_kmeans(k, tfolder, trainingSetData, parameters.useGPU)


# INFERENCING
projectionFiles = glob.glob(
    parameters.projectPath + '/Projections/*pcaModes.mat'
)
for i in range(len(projectionFiles)):
    if os.path.exists(projectionFiles[i][:-4] + '_uVals.mat'):
        print('Already done. Skipping.\n')
        continue

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

print(f'Embedding and Inference finished in {time.time() - tall} seconds!')

# SEGMENTATION
startsigma = 1.0
for k in parameters.kmeans_list:
    mmpy.findWatershedRegions(
        parameters,
        minimum_regions=k,
        startsigma=startsigma,
        pThreshold=[0.33, 0.67],
        saveplot=True,
        endident='*_pcaModes.mat'
    )
