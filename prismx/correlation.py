from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy import stats
from scipy.stats import zscore
import h5py as h5
import numpy as np
import pandas as pd
import random
from typing import List
import sys
import archs4py as a4

from prismx.utils import quantile_normalize, normalize

np.seterr(divide='ignore', invalid='ignore')

def calculate_correlation(h5_file: str, clustering: pd.DataFrame, gene_idx: List[int], cluster_id: str="global", max_sample_count: int=2000, method: str="pearson") -> pd.DataFrame:
    '''
    Returns correlation matrix for specified samples

            Parameters:
                    h5file (string): path to expression h5 file
                    clustering (pandas DataFrame): sample mappings
                    geneidx (array of int): array of gene indices

            Returns:
                    correlation coefficients (pandas DataFrame)
                    average sample correlation
    '''
    samples = np.array(a4.meta.get_meta_sample_field(h5_file,'geo_accession'))
    genes = np.array(a4.meta.get_meta_gene_field(h5_file,'gene_symbol'))

    if cluster_id == "global":
        samples_idx = sorted(random.sample(range(len(samples)), min(max_sample_count, len(samples))))
    else:
        samples_idx = np.where(clustering.loc[:,"clusterID"] == int(cluster_id))[0]
        if max_sample_count > 2: samples_idx = sorted(random.sample(set(samples_idx), min(len(samples_idx), max_sample_count)))
    
    exp = a4.data.index(h5_file, samples_idx, gene_idx=gene_idx, silent=True)
    qq = normalize(exp, transpose=False)
    del exp

    if method == "spearman":
        cc = stats.spearmanr(qq.T)[0]
    else:
        cc = np.corrcoef(qq)
    cc = np.nan_to_num(cc)
    del qq
    correlation = pd.DataFrame(cc, index=genes[gene_idx], columns=genes[gene_idx], dtype=np.float16)
    correlation.index = genes[gene_idx]
    correlation.columns = genes[gene_idx]
    del cc
    np.fill_diagonal(correlation.to_numpy(), float('nan'))
    return correlation

def create_clustering(h5_file: str, work_dir, gene_idx: List[int], gene_count: int=1000, cluster_count: int=50, deterministic: bool=True, reuse_clustering=False, method: str="minibatch") -> pd.DataFrame:
    '''
    Returns cluster association for all samples in input expression h5 file

            Parameters:
                    h5file (string): path to expression h5 file
                    geneIndices (array type int): indices of genes
                    geneCount (int) count of genes used for clustering
                    clusterCount (int): number of clusters
                    method (str): clustering method minibatch/kmeans (default: minibatch)
            Returns:
                    sample cluster mapping (pandas.DataFrame)
    '''
    if deterministic:
        random.seed(42)
    
    if reuse_clustering:
        try:
            cluster_mapping = pd.read_csv(work_dir+"/clustering.tsv", sep="\t")
            cluster_mapping.index = cluster_mapping.iloc[:,0]
            cluster_mapping.columns=["sampleID", "clusterID"]
            if len(set(cluster_mapping.iloc[:,1])) == cluster_count:
                return cluster_mapping
        except Exception:
            x = "file could not be read or clustering number does not match"

    samples = a4.meta.get_meta_sample_field(h5_file,'geo_accession')

    exp = a4.data.index(h5_file, list(range(len(samples))), gene_idx=sorted(random.sample(gene_idx, gene_count)))

    qq = normalize(exp, transpose=False)
    qq = pd.DataFrame(zscore(qq, axis=1)).fillna(0)
    exp = None

    clustering = [] 
    if method == "minibatch":
        clustering = MiniBatchKMeans(init ='k-means++',
                        n_clusters = cluster_count,
                        batch_size = 2500,
                        n_init = 10,
                        max_no_improvement = 500).fit(qq.transpose()).labels_
    else:
        clustering = KMeans(n_clusters=cluster_count, random_state=42).fit(qq.transpose()).labels_
    qq = None     # keep memory footprint low

    cluster_mapping = pd.DataFrame({'sampleID': samples, 'clusterID': clustering}, index = samples, columns=["sampleID", "clusterID"])
    return cluster_mapping
