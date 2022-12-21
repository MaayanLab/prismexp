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
from prismx.correlation import create_clustering, calculate_correlation
from prismx.feature import features
from prismx.training import train
from prismx.utils import get_config, help, read_gmt, normalize
from prismx.loaddata import list_libraries, download_expression, load_library, print_libraries, get_genes
from prismx.prediction import predict, prismx_predictions
from prismx.validation import benchmark_gmt, benchmarkGMTfast, benchmark_gmt_fast
from prismx.bridgegsea import bridge_gsea, plot_enrichment, plot_gsea, nes

def create_correlation_matrices(h5_file: str, work_dir: str, cluster_count: int=50, read_threshold: int=20, sample_threshold: float=0.01, filter_samples: int=2000, correlation_matrix_count: int=50, cluster_method: str="minibatch", cluster_gene_count: int=1000, sample_count: int=5000, reuse_clustering: bool=False, correlation_method: str="pearson", verbose: bool=True):
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
    os.makedirs(work_dir, exist_ok=True)
    if verbose: print("1. Filter genes")
    tstart = time.time()
    filtered_genes = filterGenes(h5_file, read_threshold, sample_threshold, filter_samples)
    elapsed = round((time.time()-tstart)/60, 2)
    if verbose: print("   -> completed in "+str(elapsed)+"min / #genes="+str(len(filtered_genes)))
    if verbose: print("2. Cluster samples")
    tstart = time.time()
    clustering = create_clustering(h5_file, work_dir, filtered_genes, cluster_gene_count, cluster_count, reuse_clustering=reuse_clustering, method=cluster_method)
    tclust = clustering.iloc[:,1]
    tclust.to_csv(work_dir+"/clustering.tsv", sep="\t")
    elapsed = round((time.time()-tstart)/60,2)
    if verbose: print("   -> completed in "+str(elapsed)+"min")
    if verbose: print("3. Calculate "+str(cluster_count)+" correlation matrices")
    tstart = time.time()
    mats = list(range(cluster_count))
    mats.append("global")
    j = 0
    os.makedirs(work_dir+"/correlation", exist_ok=True)
    if verbose: bar = Bar('Processing correlation', max=len(mats))
    for i in range(0, len(mats)):
        cor_mat = calculate_correlation(h5_file, clustering, filtered_genes, cluster_id=mats[i], max_sample_count=sample_count, method=correlation_method)
        cor_mat.columns = cor_mat.columns.astype(str)
        cor_mat.reset_index().to_feather(work_dir+"/correlation/correlation_"+str(mats[i])+".f")
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
