from .setrunparameters import setRunParameters
from .mmutils import findPointDensity, gencmap, createProjectDirectory
from .motionmapper import run_tSne, run_UMAP, runEmbeddingSubSampling, subsampled_tsne_from_projections, findEmbeddings, set_kmeans_model, findClusters
from .wavelet import findWavelets
from .wshed import findWatershedRegions
from .demoutils import makeregionvideo_flies, getTransitions, makeTransitionMatrix, doTheShannonShuffle, \
    plotLaggedEigenvalues, makeregionvideos_mice