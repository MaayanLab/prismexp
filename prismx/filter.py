import pandas as pd
import h5py as h5
from typing import List
import random
import numpy as np
from sklearn.cluster import KMeans

from prismx.utils import normalize
import archs4py as a4


def filterGenes(h5file: str, readThreshold: int=20, sampleThreshold: float=0.02, filterSamples: int=2000, deterministic: bool=True) -> List[int]:
    '''
    Returns filtered genes with sufficient read support
        Parameters:
                h5file          (string): path to expression h5 file
                readThreshold      (int): minimum number of reads required for gene filtering
                sampleThreshold  (float): fraction of samples required with read count larger than _readThreshold
                filterSamples      (int): number of samples used to identify genes for clustering

        Returns:
                (List[int]): filtered index of genes passing criteria
    '''
    if deterministic:
        random.seed(42)

    exp = a4.data.rand(h5file, 100, filterSingle=True)
    #f = h5.File(h5file, 'r')
    #expression = f['data/expression']
    #filterSamples = min(expression.shape[1], filterSamples)
    #random_samples = sorted(random.sample(range(expression.shape[1]), filterSamples))
    #exp = pd.DataFrame(expression[:, random_samples], dtype=np.int32)
    kk = exp[exp > readThreshold].count(axis=1)
    del exp
    #f.close()
    return([idx for idx, val in enumerate(kk) if val >= exp.shape[1]*sampleThreshold])

def geneClustering(h5file: str, geneidx: List[int], clusterCount: int=100, sampleCount: int=3000) -> pd.DataFrame:
    '''
    Returns cluster association for all genes in input expression h5 file

        Parameters:
                h5file      (string): path to expression h5 file
                geneidx  (List[int]): indices of genes
                clusterCount   (int): number of clusters
        Returns:
                (pandas.DataFrame): gene cluster mapping
    '''
    f = h5.File(h5file, 'r')
    expression = f['data/expression']
    samples = f['meta/samples/geo_accession']
    genes = f['meta/genes/gene_symbol']
    sampleCount = min(len(samples), sampleCount)
    clusterCount = min(len(genes), clusterCount)
    sampleidx = random.sample(range(0, len(samples)), sampleCount)
    sampleidx.sort()
    geneidx.sort()
    exp = 0     # keep memory footprint low
    exp = expression[geneidx,:][:, sampleidx]
    qq = normalize(exp, transpose=False)
    kmeans = KMeans(n_clusters=clusterCount, random_state=42).fit(qq)
    qq = 0      # keep memory footprint low
    clustering = kmeans.labels_
    kmeans = 0
    clusterMapping = pd.DataFrame({'geneID': geneidx, 'clusterID': clustering}, index = geneidx, columns=["geneID", "clusterID"])
    return(clusterMapping)
