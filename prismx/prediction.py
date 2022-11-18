import pandas as pd
import numpy as np
import os
import math
import random
import pickle
import time
import multiprocessing
from typing import List
from progress.bar import Bar
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import zscore

from prismx.utils import read_gmt, load_correlation, load_feature
from prismx.feature import features, load_features_range

def predict(workdir: str, gmt_file: str, model=0, step_size: int=1000, intersect: bool=False, normalize:bool=False, verbose: bool=False):
    if model == 0:
        model = pickle.load(open(workdir+"/model.pkl", "rb"))
    features(gmt_file, workdir, intersect=intersect, verbose=verbose)
    prismx_predictions(model, workdir, os.path.basename(gmt_file), step_size, normalize=normalize, verbose=verbose)

def prismx_predictions(model, workdir: str, prediction_name: str, step_size: int=1000, verbose: bool=False, normalize=False):
    os.makedirs(workdir+"/predictions", exist_ok=True)
    prediction_size = load_feature(workdir, 0).shape[1]
    prism = pd.DataFrame()
    step_number = math.ceil(prediction_size/step_size)
    #if verbose: bar = Bar('Processing predictions', max=step_number)
    
    for i in tqdm.tqdm(range(0, step_number), desc="Load Features", disable=(not verbose)):
        rfrom = i*step_size
        rto = min((i+1)*step_size, prediction_size)
        features = load_features_range(workdir, rfrom, rto)
        prism = make_predictions_range(model, prism, features)
        features = 0
        #if verbose: bar.next()
    #if verbose: bar.finish()
    if normalize:
        prism = prism.apply(zscore)
    prism.reset_index().to_feather(workdir+"/predictions/"+prediction_name+".f")

def make_predictions_range(model: str, prism: pd.DataFrame, features: List[pd.DataFrame], verbose: bool=False) -> pd.DataFrame:
    pred_list = []
    for i in range(0, features[0].shape[1]):
        start = time.time()
        df = []
        k = 0
        for ff in features:
            df.append(ff.iloc[:,i])
            k = k + 1
        if verbose:
            print(str(i) + " - " + str(round(time.time()-start)))
        df = pd.DataFrame(np.array(df).T)
        df.fillna(0, inplace=True)
        pred_list.append(model.predict(df))
    prism_temp = pd.DataFrame(pred_list).transpose()
    prism_temp.columns = features[0].columns
    prism_temp.index = features[0].index
    if prism.shape[1] == 0:
        prism = prism_temp
    else:
        prism = pd.concat((prism, prism_temp), axis=1)
    return(prism)
