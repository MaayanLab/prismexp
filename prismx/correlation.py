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
    exp = None

    if method == "spearman":
        cc = stats.spearmanr(qq.T)[0]
    else:
        cc = np.corrcoef(qq)
    cc = np.nan_to_num(cc)
    qq = None
    correlation = pd.DataFrame(cc, index=genes[gene_idx], columns=genes[gene_idx], dtype=np.float16)
    correlation.index = genes[gene_idx]
    correlation.columns = genes[gene_idx]
    cc = None
    np.fill_diagonal(correlation.to_numpy(), float('nan'))
    return correlation

def create_clustering(work_dir: str, h5_file: str, gene_idx: List[int], gene_count: int=1000, cluster_count: int=50, deterministic: bool=True, random_state: int=1, min_reads: int=2000, reuse_clustering=False, method: str="minibatch") -> pd.DataFrame:
    """
    Returns cluster association for all samples in input expression h5 file.
    
    Parameters:
        work_dir (str): path to directory where clustering results will be stored
        h5_file (str): path to expression h5 file
        gene_idx (List[int]): indices of genes to use for clustering
        gene_count (int): count of genes to use for clustering (default: 1000)
        cluster_count (int): number of clusters to generate (default: 50)
        min_reads (int): minimu number of reads for a sample to be considered in clustering
        deterministic (bool): whether to set the random seed to ensure reproducibility (default: True)
        random_state (int): random seed to use (default: 1)
        reuse_clustering (bool): whether to reuse previous clustering results stored in work_dir if available (default: False)
        method (str): clustering method to use, either "minibatch" or "kmeans" (default: "minibatch")
    
    Returns:
        sample cluster mapping (pandas.DataFrame)
    """
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
    filtered_sample_idx = np.array(np.where(np.sum(exp, axis=0) > min_reads)[0])
    exp = exp.iloc[:, filtered_sample_idx]
    print("Number of samples used in clustering:", exp.shape[1])

    qq = normalize(exp, transpose=False)
    qq = pd.DataFrame(zscore(qq, axis=1)).fillna(0)
    exp = None

    clustering = [] 
    if method == "minibatch":
        clustering = MiniBatchKMeans(init ='k-means++',
                        n_clusters = cluster_count,
                        batch_size = 2500,
                        n_init = 10,
                        random_state=random_state,
                        max_no_improvement = 500).fit(qq.transpose()).labels_
    else:
        clustering = KMeans(n_clusters=cluster_count, random_state=42).fit(qq.transpose()).labels_
    qq = None     # keep memory footprint low

    cluster_mapping = pd.DataFrame({'sampleID': np.array(samples)[filtered_sample_idx], 'clusterID': clustering}, index = np.array(samples)[filtered_sample_idx], columns=["sampleID", "clusterID"])
    return cluster_mapping
