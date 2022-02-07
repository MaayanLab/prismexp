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

from importlib import reload
reload(prismx.bridgegsea)

from prismx.filter import filterGenes
from prismx.correlation import createClustering, calculateCorrelation
from prismx.feature import features
from prismx.training import train
from prismx.utils import get_config, help, read_gmt, normalize
from prismx.loaddata import list_libraries, download_expression, load_library, print_libraries, get_genes
from prismx.prediction import predict, prismx_predictions
from prismx.validation import benchmark_gmt, benchmarkGMTfast, benchmark_gmt_fast
from prismx.bridgegsea import bridge_gsea, plot_enrichment, plot_gsea, nes

def create_correlation_matrices(h5file: str, outputFolder: str, clusterCount: int=50, readThreshold: int=20, sampleThreshold: float=0.01, filterSamples: int=2000, correlationMatrixCount: int=50, clusterGeneCount: int=1000, sampleCount: int=5000, verbose: bool=True):
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
    filtered_genes = filterGenes(h5file, readThreshold, sampleThreshold, filterSamples)
    elapsed = round((time.time()-tstart)/60, 2)
    if verbose: print("   -> completed in "+str(elapsed)+"min / #genes="+str(len(filtered_genes)))
    if verbose: print("2. Cluster samples")
    tstart = time.time()
    clustering = createClustering(h5file, filtered_genes, clusterGeneCount, clusterCount)
    tclust = clustering.iloc[:,1]
    tclust.index = [x.decode("UTF-8") for x in tclust.index]
    tclust.to_csv(outputFolder+"/clustering.tsv", sep="\t")
    elapsed = round((time.time()-tstart)/60,2)
    if verbose: print("   -> completed in "+str(elapsed)+"min")
    if verbose: print("3. Calculate "+str(clusterCount)+" correlation matrices")
    tstart = time.time()
    mats = list(range(clusterCount))
    mats.append("global")
    j = 0
    os.makedirs(outputFolder+"/correlation", exist_ok=True)
    if verbose: bar = Bar('Processing correlation', max=len(mats))
    for i in range(0, len(mats)):
        cor_mat = calculateCorrelation(h5file, clustering, filtered_genes, clusterID=mats[i], maxSampleCount=sampleCount)
        cor_mat.columns = cor_mat.columns.astype(str)
        cor_mat.reset_index().to_feather(outputFolder+"/correlation/correlation_"+str(mats[i])+".f")
        j = j+1
        if verbose: bar.next()
    elapsed = round((time.time()-tstart)/60,2)
    if verbose: print("   -> completed in "+str(elapsed)+"min")
    if verbose: bar.finish()

def test_data() -> str:
    path = os.path.join(
        os.path.dirname(__file__),
        'data/expression_sample.h5'
    )
    return(path)
