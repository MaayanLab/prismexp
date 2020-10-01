import pandas as pd
import h5py as h5
from typing import List
import random
import numpy as np
from sklearn.cluster import KMeans

from prismx.utils import quantile_normalize, normalize


def filterGenes(h5file: str, readThreshold: int=20, sampleThreshold: float=0.01, filterSamples: int=5000) -> List[int]:
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
    f = h5.File(h5file, 'r')
    expression = f['data/expression']
    genes = f['meta/genes']
    samples = f['meta/Sample_geo_accession']
    filterSamples = min(len(samples), filterSamples)
    rsamples = random.sample(range(0, len(samples)), filterSamples)
    rsamples.sort()
    exp = pd.DataFrame(expression[rsamples, :])
    kk = exp[exp > readThreshold].count()
    exp = 0
    expression = 0
    samples = 0
    genes = 0
    f.close()
    filteredGenes = [idx for idx, val in enumerate(kk) if val >= len(rsamples)*sampleThreshold]
    return(filteredGenes)

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
    samples = f['meta/Sample_geo_accession']
    genes = f['meta/genes']
    sampleCount = min(len(samples), sampleCount)
    clusterCount = min(len(genes), clusterCount)
    sampleidx = random.sample(range(0, len(samples)), sampleCount)
    sampleidx.sort()
    geneidx.sort()
    exp = 0     # keep memory footprint low
    exp = expression[sampleidx,:][:, geneidx]
    qq = normalize(exp, stepSize=500, transpose=True)
    kmeans = KMeans(n_clusters=clusterCount, random_state=42).fit(qq)
    qq = 0      # keep memory footprint low
    clustering = kmeans.labels_
    kmeans = 0
    clusterMapping = pd.DataFrame({'geneID': geneidx, 'clusterID': clustering}, index = geneidx, columns=["geneID", "clusterID"])
    return(clusterMapping)

def hykGeneSelection(h5file: str, geneidx: List[int], geneCount: int=500, clusterCount: int=500, sampleCount: int=3000) -> List[int]:
    '''
    Returns a set of genes with uncorrelated gene expression

            Parameters:
                    h5file (string): path to expression h5 file
                    geneidx (array type int): indices of genes
                    geneCount (int): number of genes to be selected
                    clusterCount (int): number of clusters
                    sampleCount (int): number of samples used for correlation
            Returns:
                    gene cluster mapping (pandas.DataFrame)
    '''
    f = h5.File(h5file, 'r')
    genes = f['meta/genes']
    samples = f['meta/Sample_geo_accession']
    clusterCount = min(len(genes), clusterCount)
    geneCount = min(len(genes), geneCount)
    clusterMapping = geneClustering(h5file, geneidx, clusterCount, sampleCount)
    selectedGenes = set()
    while len(selectedGenes) < geneCount:
        genes = np.where(clusterMapping.loc[:,"clusterID"] == len(selectedGenes)%clusterCount)[0]
        selectedGenes.add(random.sample(set(genes),1)[0])
    selectedGenes = list(selectedGenes)
    selectedGenes.sort()
    return(selectedGenes)