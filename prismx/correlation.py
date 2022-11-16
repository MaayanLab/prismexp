from sklearn.cluster import KMeans
from scipy.stats import zscore
import h5py as h5
import numpy as np
import pandas as pd
import random
from typing import List
import sys

from prismx.utils import quantile_normalize, normalize

np.seterr(divide='ignore', invalid='ignore')

def calculateCorrelation(h5file: str, clustering: pd.DataFrame, geneidx: List[int], clusterID: str="global", maxSampleCount: int=2000) -> pd.DataFrame:
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
    f = h5.File(h5file, 'r')
    samples = f['meta/samples/geo_accession']
    genes = f['meta/genes/gene_symbol']
    if clusterID == "global":
        samplesidx = sorted(random.sample(range(len(samples)), min(maxSampleCount, len(samples))))
    else:
        samplesidx = np.where(clustering.loc[:,"clusterID"] == int(clusterID))[0]
        if maxSampleCount > 2: samplesidx = sorted(random.sample(set(samplesidx), min(len(samplesidx), maxSampleCount)))
    exp = f['data/expression'][:,samplesidx][geneidx,:]
    qq = normalize(exp, transpose=False)
    exp = 0
    cc = np.corrcoef(qq)
    cc = np.nan_to_num(cc)
    qq = 0
    correlation = pd.DataFrame(cc, index=genes[geneidx], columns=genes[geneidx], dtype=np.float16)
    correlation.index = [x.upper() for x in genes[geneidx]]
    correlation.columns = [x.upper() for x in genes[geneidx]]
    f.close()
    cc = 0
    np.fill_diagonal(correlation.to_numpy(), float('nan'))
    return(correlation)

def createClustering(h5file: str, geneidx: List[int], geneCount: int=500, clusterCount: int=50, deterministic: bool=True) -> pd.DataFrame:
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
    f = h5.File(h5file, 'r')
    expression = f['data/expression']
    samples = list(f['meta/samples/geo_accession'])
    genes = sorted(random.sample(geneidx, geneCount))
    exp = expression[genes, :]
    f.close()
    qq = normalize(exp, transpose=False)
    qq = pd.DataFrame(zscore(qq, axis=1)).fillna(0)
    exp = 0
    kmeans = KMeans(n_clusters=clusterCount, random_state=42).fit(qq.transpose())
    qq = 0      # keep memory footprint low
    clustering = kmeans.labels_
    kmeans = 0  # keep memory footprint low
    clusterMapping = pd.DataFrame({'sampleID': samples, 'clusterID': clustering}, index = samples, columns=["sampleID", "clusterID"])
    return(clusterMapping)
