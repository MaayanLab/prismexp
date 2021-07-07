from typing import Dict, List
import time
import pandas as pd
import numpy as np
import os
from progress.bar import Bar
import multiprocessing

from prismx.utils import read_gmt, load_correlation, load_feature
from prismx.loaddata import get_genes


def correlation_scores(gmt_file: str, workdir: str, intersect: bool=False, threads: int=4, verbose: bool=False):
    os.makedirs(workdir+"/features", exist_ok=True)
    correlation_files = os.listdir(workdir+"/correlation")
    cct = load_correlation(workdir, 0)
    background_genes = cct.columns
    cct = 0
    ugenes = []
    library, rev_library, unique_genes = read_gmt(gmt_file, background_genes, verbose=verbose)
    if intersect:
        ugenes = list(set(sum(library.values(), [])))
        ugenes = list(set(ugenes) & set(background_genes))
        ugenes = [x.encode("UTF-8") for x in ugenes]
        if verbose:
            print("overlapping genes: "+str(len(ugenes)))
    lk = list(range(0, len(correlation_files)-1))
    lk.append("global")
    if verbose: bar = Bar('Processing average correlation', max=len(lk))
    params = list()
    for ll in lk:
        params.append((workdir, ll, library, intersect, ugenes))
    process_pool = multiprocessing.Pool(threads)
    process_pool.starmap(get_average_correlation, params)
    if verbose: bar.finish()

def get_average_correlation(workdir: str, i: int, library: Dict, intersect: bool=False, ugenes: List=[]):
    correlation = load_correlation(workdir, i)
    preds = []
    for ll in list(library.keys()):
        if intersect:
            preds.append(correlation.loc[:, library[ll]].loc[ugenes,:].mean(axis=1))
        else:
            preds.append(correlation.loc[:, library[ll]].mean(axis=1))
    features = pd.concat(preds, axis=1)
    features.columns = list(library.keys())
    features = pd.DataFrame(features.fillna(0), dtype=np.float16)
    features = features.reset_index()
    features.columns = features.columns.astype(str)
    features.to_feather(workdir+"/features/features_"+str(i)+".f")

def load_features(workdir: str, verbose: bool=False): 
    features = []
    feature_files = os.listdir(workdir+"/features")
    lk = list(range(0, len(feature_files)-1))
    lk.append("global")
    for i in lk:
        feature = load_feature(workdir,i)
        features.append(feature)
        if verbose:
            print("features_"+str(i)+".f")
    return(features)

def load_features_range(workdir: str, range_from: int, range_to: int, verbose: bool=False): 
    features = []
    feature_files = os.listdir(workdir+"/features")
    lk = list(range(0, len(feature_files)-1))
    lk.append("global")
    for i in lk:
        feature = load_feature(workdir, i)
        features.append(feature.iloc[:, range_from:range_to].copy())
        feature = 0
        if verbose:
            print("features_"+str(i)+".f")
    return(features)
