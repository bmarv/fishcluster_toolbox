import glob
import math
import os
import numpy as np
import bisect
from config import projectPath
import motionmapperpy as mmpy

def pointsInCircum(r,n=100):
    return [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in range(0,n+1)]

def runs_of_ones_array(bits):
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts, run_starts, run_ends

def combine_ones_and_zeros(ones, zeros, th, size):
    block = 0
    hits = 0
    records=list()
    j,i = 0,0
    while i < ones.shape[0]:
        if block >= size:
            if (hits/block)>=th:
                records.append({'idx':(j,i), 'score':(hits/block)})
                block,hits,j = 0,0,i
            else:
                block -= (ones[j] + zeros[j])
                hits -= ones[j]
                j+=1
        if block == 0:
            block, hits = ones[i], ones[i]
            i+=1
        elif block < size:
            block += (ones[i] + zeros[i-1])
            hits += ones[i]
            i+=1
    return records

def get_cluster_sequences(clusters, cluster_ids=range(1,6), sw=6*60*5, th=0.6):
    records = dict(zip(cluster_ids, [list() for i in range(len(cluster_ids))]))
    for cid in cluster_ids:
        bits = clusters==cid
        n_ones, rs,re = runs_of_ones_array(bits)
        n_zeros = rs[1:] - re[:-1]
        matches = combine_ones_and_zeros(n_ones, n_zeros, th, sw)
        matches.sort(key=lambda x: x["score"], reverse=True)
        results = [(rs[m['idx'][0]],re[m['idx'][1]], m['score']) for m in matches]
        records[cid].extend(results)
    return records

def split_into_batches(time, data, batch_size=60*60*5, flt=True):
    """
    @param time: time vector in data frames 

    Split data into batches of batch_size dataframes (5 df per second)
    flt: if True, filter out batches with less than 1/4 of the batch_size
    """
    t = time-time[0]
    step = int(batch_size) # 5 df per second
    hours_end = [bisect.bisect_left(t, h) for h in range(step,int(t[-1]), step)]
    batches = np.split(data, hours_end)
    times = np.split(time, hours_end)
    batches = list(filter(lambda v: len(v)>0, batches))
    times = list(filter(lambda v: len(v)>0, times))
    return times, batches

def get_individuals_keys(parameters, block=""):
    files = glob.glob(parameters.projectPath+f"/Projections/{block}*_pcaModes.mat")
    return sorted(list(set(map(lambda f: "_".join(f.split("/")[-1].split("_")[:3]),files))))

def get_days(parameters, prefix=""):
    files = glob.glob(parameters.projectPath+f"/Projections/{prefix}*_pcaModes.mat")
    return sorted(list(set(map(lambda f: "_".join(f.split("/")[-1].split("_")[3:5]),files))))

def set_parameters(parameters=None): 
    parameters = mmpy.setRunParameters(parameters)
    parameters.pcaModes = 3
    parameters.samplingFreq = 5
    parameters.maxF = 2.5
    parameters.minF = 0.01
    parameters.omega0 = 5
    parameters.numProcessors = 16
    parameters.method="UMAP"
    parameters.kmeans = 10
    parameters.kmeans_list = [5, 7, 10, 20, 50, 100]
    parameters.projectPath = projectPath
    os.makedirs(parameters.projectPath,exist_ok=True)
    mmpy.createProjectDirectory(parameters.projectPath)
    return parameters