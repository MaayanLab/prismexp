from sklearn.cluster import KMeans
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

def calculate_correlation(h5file: str, clustering: pd.DataFrame, geneidx: List[int], clusterID: str="global", maxSampleCount: int=2000) -> pd.DataFrame:
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
    samples = np.array(a4.meta.get_meta_sample_field(h5file,'geo_accession'))
    genes = np.array(a4.meta.get_meta_gene_field(h5file,'gene_symbol'))

    if clusterID == "global":
        samplesidx = sorted(random.sample(range(len(samples)), min(maxSampleCount, len(samples))))
    else:
        samplesidx = np.where(clustering.loc[:,"clusterID"] == int(clusterID))[0]
        if maxSampleCount > 2: samplesidx = sorted(random.sample(set(samplesidx), min(len(samplesidx), maxSampleCount)))
    
    exp = a4.data.index(h5file, samplesidx, gene_idx=geneidx, silent=True)
    qq = normalize(exp, transpose=False)
    exp = 0
    cc = np.corrcoef(qq)
    cc = np.nan_to_num(cc)
    qq = 0
    correlation = pd.DataFrame(cc, index=genes[geneidx], columns=genes[geneidx], dtype=np.float16)
    correlation.index = genes[geneidx]
    correlation.columns = genes[geneidx]
    cc = 0
    np.fill_diagonal(correlation.to_numpy(), float('nan'))
    return correlation

def create_clustering(h5file: str, workdir, geneidx: List[int], geneCount: int=500, clusterCount: int=50, deterministic: bool=True, reuseClustering=False) -> pd.DataFrame:
    '''
    Returns cluster association for all samples in input expression h5 file

            Parameters:
                    h5file (string): path to expression h5 file
                    geneIndices (array type int): indices of genes
                    geneCount (int) count of genes used for clustering
                    clusterCount (int): number of clusters
            Returns:
                    sample cluster mapping (pandas.DataFrame)
    '''
    if deterministic:
        random.seed(42)
    
    if reuseClustering:
        try:
            clusterMapping = pd.read_csv(workdir+"/clustering.tsv")
            clusterMapping.index = clusterMapping.iloc[:,0]
            clusterMapping.columns=["sampleID", "clusterID"]
            if len(set(clusterMapping.iloc[:,1])) == clusterCount:
                return clusterMapping
        except Exception:
            x = "file could not be read or clustering number does not match"

    samples = a4.meta.get_meta_sample_field(h5file,'geo_accession')

    exp = a4.data.index(h5file, list(range(len(samples))), gene_idx=sorted(random.sample(geneidx, geneCount)))

    qq = normalize(exp, transpose=False)
    qq = pd.DataFrame(zscore(qq, axis=1)).fillna(0)
    exp = 0
    kmeans = KMeans(n_clusters=clusterCount, random_state=42).fit(qq.transpose())
    qq = 0      # keep memory footprint low
    clustering = kmeans.labels_
    kmeans = 0  # keep memory footprint low
    clusterMapping = pd.DataFrame({'sampleID': samples, 'clusterID': clustering}, index = samples, columns=["sampleID", "clusterID"])
    return clusterMapping
