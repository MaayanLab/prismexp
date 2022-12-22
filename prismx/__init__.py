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
import tqdm

from prismx.filter import filterGenes
from prismx.correlation import create_clustering, calculate_correlation
from prismx.feature import features
from prismx.training import train
from prismx.utils import get_config, help, read_gmt, normalize
from prismx.loaddata import list_libraries, download_expression, load_library, print_libraries, get_genes
from prismx.prediction import predict, prismx_predictions
from prismx.validation import benchmark_gmt, benchmarkGMTfast, benchmark_gmt_fast
from prismx.bridgegsea import bridge_gsea, plot_enrichment, plot_gsea, nes

def create_correlation_matrices(h5_file: str, work_dir: str, cluster_count: int=100, read_threshold: int=20, sample_threshold: float=0.01, filter_samples: int=2000, min_avg_reads_per_gene: int=2, cluster_method: str="minibatch", cluster_gene_count: int=1000, sample_count: int=5000, reuse_clustering: bool=False, correlation_method: str="pearson", verbose: bool=True):
    """
    Calculate clustering and correlation matrices for the samples in the specified h5 file.

    Parameters
    ----------
    h5_file : str
        The path to the h5 file containing the gene expression data.
    work_dir : str
        The directory to save the resulting clustering and correlation matrices.
    cluster_count : int, optional
        The number of clusters to use for the sample clustering. Default is 100.
    read_threshold : int, optional
        The minimum number of reads a gene must have in a fraction of total reads to keep. Default is 20.
    sample_threshold : float, optional
        The minimum fraction of samples that contain read_threshold reads of a gene to keep. Default is 0.01.
    filter_samples : int, optional
        The maximum number of samples to use for gene filtering. Default is 2000.
    min_avg_reads_per_gene : int, optional
        The average number of reads per gene for a sample to be considered in the clustering
    cluster_method : str, optional
        The clustering method to use. Options are "minibatch" and "kmeans". Default is "minibatch".
    cluster_gene_count : int, optional
        The number of genes to use for the sample clustering. Default is 1000.
    sample_count : int, optional
        The maximum number of samples to use for calculating the correlation matrices. Default is 5000.
    reuse_clustering : bool, optional
        Whether to reuse the existing clustering results in the work directory. Default is False.
    correlation_method : str, optional
        The correlation method to use. Options are "pearson" and "spearman". Default is "pearson".
    verbose : bool, optional
        Whether to print progress messages. Default is True.

    Returns
    -------
    None
        The resulting clustering and correlation matrices are saved in the specified work directory.
    """
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
    for i in tqdm.tqdm(range(0, len(mats)), disable=not verbose):
        cor_mat = calculate_correlation(h5_file, clustering, filtered_genes, cluster_id=mats[i], max_sample_count=sample_count, method=correlation_method)
        cor_mat.columns = cor_mat.columns.astype(str)
        cor_mat.reset_index().to_feather(work_dir+"/correlation/correlation_"+str(mats[i])+".f")
        j = j+1

    elapsed = round((time.time()-tstart)/60,2)
    if verbose: print("   -> completed in "+str(elapsed)+"min")

def test_data() -> str:
    path = os.path.join(
        os.path.dirname(__file__),
        'data/expression_sample.h5'
    )
    return(path)
