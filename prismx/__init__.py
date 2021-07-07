from sklearn.cluster import KMeans
import h5py as h5
import numpy as np
import pandas as pd
import feather
import time
import pickle
import os
from progress.bar import Bar
from typing import List

from prismx.filter import filterGenes
from prismx.correlation import createClustering, calculateCorrelation
from prismx.prediction import correlationScores
from prismx.training import trainModel
from prismx.utils import getConfig, help, readGMT, normalize
from prismx.loaddata import listLibraries, loadExpression, loadLibrary, printLibraries, getGenes
from prismx.prismxprediction import predictGMT, prismxPredictions
from prismx.validation import benchmarkGMT, benchmarkGMTfast, benchmarkGMTfastPx

def createCorrelationMatrices(h5file: str, outputFolder: str, clusterCount: int=50, readThreshold: int=20, sampleThreshold: float=0.01, filterSamples: int=2000, correlationMatrixCount: int=50, clusterGeneCount: int=1000, correlationSampleCount: int=5000, verbose: bool=True):
    '''
    Write a set of correlation matrices, by partitioning gene expression into clusters and applying Pearson
    correlation for pairs of genes. It will also create an additional matrix for global correlation.

            Parameters:
                    h5file (string): path to expression h5 file
                    geneidx (array type int): indices of genes
                    geneCount (int): number of genes to be selected
                    clusterCount (int): number of clusters
                    sampleCount (int): number of samples used for correlation
            Returns:
                    gene cluster mapping (pandas.DataFrame)
    '''
    if verbose: print("1. Filter genes")
    tstart = time.time()
    filteredGenes = filterGenes(h5file, readThreshold, sampleThreshold, filterSamples)
    elapsed = round((time.time()-tstart)/60, 2)
    if verbose: print("   -> completed in "+str(elapsed)+"min / #genes="+str(len(filteredGenes)))
    if verbose: print("2. Cluster samples")
    tstart = time.time()
    clustering = createClustering(h5file, filteredGenes, clusterGeneCount, clusterCount)
    pickle.dump(clustering, open(outputFolder+"/clustering.pkl", "wb"))
    elapsed = round((time.time()-tstart)/60,2)
    if verbose: print("   -> completed in "+str(elapsed)+"min")
    if verbose: print("3. Calcualate "+str(clusterCount)+" correlation matrices")
    tstart = time.time()
    mats = list(range(clusterCount))
    mats.append("global")
    j = 0
    os.makedirs(outputFolder+"/correlation", exist_ok=True)
    if verbose: bar = Bar('Processing correlation', max=len(mats))
    for i in range(0, len(mats)):
        cor_mat = calculateCorrelation(h5file, clustering, filteredGenes, clusterID=mats[i], maxSampleCount=correlationSampleCount)
        cor_mat.columns = cor_mat.columns.astype(str)
        cor_mat.reset_index().to_feather(outputFolder+"/correlation/correlation_"+str(mats[i])+".f")
        j = j+1
        if verbose: bar.next()
    elapsed = round((time.time()-tstart)/60,2)
    if verbose: print("   -> completed in "+str(elapsed)+"min")
    if verbose: bar.finish()

def testData() -> str:
    path = os.path.join(
        os.path.dirname(__file__),
        'data/expression_sample.h5'
    )
    return(path)
