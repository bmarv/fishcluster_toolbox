#!/bin/bash



mkdir /mnt/Projections; cp /mnt/raw_Projections/* /mnt/Projections/; 
python train.py --n_neighbors 50 --min_dist 0.5 --device 0|| true; 
mkdir /mnt/training_gpu_nn50_mindist0.5; mv /mnt/Projections/ /mnt/UMAP/ /mnt/training_gpu_nn50_mindist0.5/; 

mkdir /mnt/Projections; cp /mnt/raw_Projections/* /mnt/Projections/; 
python train.py --n_neighbors 15 --min_dist 0.25 --device 0|| true; 
mkdir /mnt/training_gpu_nn15_mindist0.25; mv /mnt/Projections/ /mnt/UMAP/ /mnt/training_gpu_nn15_mindist0.25/; 

mkdir /mnt/Projections; cp /mnt/raw_Projections/* /mnt/Projections/; 
python train.py --n_neighbors 15 --min_dist 0.5 --device 0|| true; 
mkdir /mnt/training_gpu_nn15_mindist0.5; mv /mnt/Projections/ /mnt/UMAP/ /mnt/training_gpu_nn15_mindist0.5/;

mkdir /mnt/Projections; cp /mnt/raw_Projections/* /mnt/Projections/; 
python train.py --n_neighbors 50 --min_dist 0.25 --device 0|| true; 
mkdir /mnt/training_gpu_nn50_mindist0.25; mv /mnt/Projections/ /mnt/UMAP/ /mnt/training_gpu_nn50_mindist0.25/; 

mkdir /mnt/Projections; cp /mnt/raw_Projections/* /mnt/Projections/; 
python train.py --n_neighbors 100 --min_dist 0.25 --device 0|| true; 
mkdir /mnt/training_gpu_nn100_mindist0.25; mv /mnt/Projections/ /mnt/UMAP/ /mnt/training_gpu_nn100_mindist0.25/;

mkdir /mnt/Projections; cp /mnt/raw_Projections/* /mnt/Projections/; 
python train.py --n_neighbors 100 --min_dist 0.1 --device 0|| true; 
mkdir /mnt/training_gpu_nn100_mindist0.1; mv /mnt/Projections/ /mnt/UMAP/ /mnt/training_gpu_nn100_mindist0.1/; 

mkdir /mnt/Projections; cp /mnt/raw_Projections/* /mnt/Projections/; 
python train.py --n_neighbors 120 --min_dist 0.1 --device 0|| true; 
mkdir /mnt/training_gpu_nn120_mindist0.1; mv /mnt/Projections/ /mnt/UMAP/ /mnt/training_gpu_nn120_mindist0.1/; 
