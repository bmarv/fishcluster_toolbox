import hdf5storage
import pickle
import glob
import time
import os
import motionmapperpy as mmpy
from utils import train_utils

tall = time.time()
parameters = train_utils.initialize_training_parameters()
mmpy.createProjectDirectory(parameters.projectPath)
parameters.normalize_func = train_utils.return_normalization_func(parameters)
print("Subsample from projections")
trainingSetData, _ = train_utils.subsample_from_projections(parameters)
tfolder = parameters.projectPath + f'/{parameters.method}/'

# UMAP embedding for the whole dataset
'''UMAP'''
trainingEmbedding = mmpy.run_UMAP(trainingSetData, parameters)
trainingSetData[trainingSetData == 0] = 1e-12  # replace 0 with 1e-12

# calc kmeans embedding
for k in parameters.kmeans_list:
    if not os.path.exists(tfolder + f'/kmeans_{k}.pkl'):
        print(f'Initializing kmeans model with {k} clusters')
        mmpy.set_kmeans_model(k, tfolder, trainingSetData, parameters.useGPU)

# calc embedding for all individuals
''' call hierarchy:
    * for all projectionFiles
        * load embeddings
        * find kmeans clusters
        * find umap clusters
        * calc output-statistics
        * save output
'''
zValstr = 'uVals'
projectionFiles = glob.glob(
    parameters.projectPath + '/Projections/*pcaModes.mat'
)
for i in range(len(projectionFiles)):
    if os.path.exists(projectionFiles[i][:-4] + f'_{zValstr}.mat'):
        print('Already done. Skipping.\n')
        continue
    # load projections for a specific dataset
    projections = hdf5storage.loadmat(projectionFiles[i])['projections']
    # kmeans inferencing on individuals
    # TODO: refactor mmpy.findClusters -> kmeans.predict
    clusters_dict = mmpy.findClusters(projections, parameters)
    for key, value in clusters_dict.items():
        hdf5storage.write(
            data={"clusters": value, "k": int(key.split("_")[1])},
            path='/',
            truncate_existing=True,
            filename=projectionFiles[i][:-4]+'_%s.mat' % (key),
            store_python_metadata=False,
            matlab_compatible=True
        )

    # umap inferencing on individuals
        # TODO: unify computing and saving datasets/embeddings
    zValues, outputStatistics = mmpy.findEmbeddings(
        projections,
        trainingSetData,
        trainingEmbedding,
        parameters
    )

    # Save embeddings
    hdf5storage.write(
        data={'zValues': zValues},
        path='/',
        truncate_existing=True,
        filename=projectionFiles[i][:-4]+'_%s.mat' % (zValstr),
        store_python_metadata=False,
        matlab_compatible=True
    )

    # Save output statistics
    with open(
        projectionFiles[i][:-4] + f'_{zValstr}_outputStatistics.pkl',
        'wb'
    ) as hfile:
        pickle.dump(outputStatistics, hfile)

    del clusters_dict, zValues, projections, outputStatistics

print('All Embeddings Saved in %i seconds!' % (time.time() - tall))

# watershed segmentation
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
