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

def predict(work_dir: str, gmt_file: str, model=None, step_size: int=1000, intersect: bool=False, normalize:bool=False, verbose: bool=False, skip_features: bool=False, threads: int=2):
    """
    Computes the feature matrices for the given gmt file and directory, and then runs the prediction algorithm using the features.
    
    Parameters:
    work_dir (str): Path to the directory containing the correlation matrices and precomputed model.
    gmt_file (str): Path to the gmt file containing the gene set library.
    model (lightGBM model, optional): The prediction model to use. Defaults to None, which loads the model from the work_dir.
    step_size (int, optional): The number of samples to process at a time. Defaults to 1000.
    intersect (bool, optional): If True, only includes unique genes present in all gene sets in the feature matrix. Defaults to False.
    normalize (bool, optional): If True, normalizes the final prediction values using a z-score. Defaults to False.
    verbose (bool, optional): If True, prints progress information. Defaults to False.
    skip_features (bool, optional): If True, skips the feature computation step. Defaults to False.
    threads (int, optional): Number of threads to use for parallel processing. Defaults to 2.
    """
    
    if model == 0:
        model = pickle.load(open(work_dir+"/model.pkl", "rb"))
    if not skip_features:
        features(work_dir, gmt_file, intersect=intersect, threads=threads, verbose=verbose)
    prismx_predictions(model, work_dir, os.path.basename(gmt_file), step_size, normalize=normalize, verbose=verbose)

def prismx_predictions(model, work_dir: str, prediction_name: str, step_size: int=1000, verbose: bool=False, normalize=False):
    """
    Makes predictions using the given model and feature matrices, and saves the predictions to the given directory.
    
    Parameters:
    model: The model to use for making predictions.
    work_dir (str): Path to the directory containing the feature matrices.
    prediction_name (str): Name to use for the prediction file.
    step_size (int, optional): The number of samples to process at a time. Defaults to 1000.
    verbose (bool, optional): If True, prints progress information. Defaults to False.
    normalize (bool, optional): If True, normalizes the predictions before saving them using z-score. Defaults to False.
    """
    os.makedirs(work_dir+"/predictions", exist_ok=True)
    prediction_size = load_feature(work_dir, 0).shape[1]
    prism = pd.DataFrame()
    step_number = math.ceil(prediction_size/step_size)
    
    for i in tqdm(range(0, step_number), desc="Make Predictions", disable=(not verbose)):
        rfrom = i*step_size
        rto = min((i+1)*step_size, prediction_size)
        features = load_features_range(work_dir, rfrom, rto)
        prism = make_predictions_range(model, prism, features)
        features = 0

    if normalize:
        prism = prism.apply(zscore)
    prism.reset_index().to_feather(work_dir+"/predictions/"+prediction_name+".f")


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
