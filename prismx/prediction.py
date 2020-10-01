from typing import Dict, List
import time
import pandas as pd
import numpy as np
import os
from progress.bar import Bar

from prismx.utils import readGMT
from prismx.loaddata import getGenes

def correlationScores(gmtFile: str, correlationFolder: str, outFolder: str, intersect: bool=False, verbose: bool=False):
    os.makedirs(outFolder, exist_ok=True)
    correlation_files = os.listdir(correlationFolder)
    cct = pd.read_feather(correlationFolder+"/"+correlation_files[0])
    backgroundGenes = [x.upper() for x in cct.columns]
    cct = 0
    ugenes = []
    library, revLibrary, uniqueGenes = readGMT(gmtFile, backgroundGenes, verbose=verbose)
    if intersect:
        ugenes = list(set(sum(library.values(), [])))
        ugenes = list(set(ugenes) & set(getGenes(correlationFolder)))
        if verbose:
            print("overlapping genes: "+str(len(ugenes)))
    lk = list(range(0, len(correlation_files)-1))
    lk.append("global")
    if verbose: bar = Bar('Processing average correlation', max=len(lk))
    for i in lk:
        getAverageCorrelation(correlationFolder, i, library, outFolder, intersect=intersect, ugenes = ugenes)
        if verbose: bar.next()
    if verbose: bar.finish()

def getAverageCorrelation(correlationFolder: str, i: int, library: Dict, outFolder: str, intersect: bool=False, ugenes: List=[]):
    correlation = loadCorrelation(correlationFolder, i)
    preds = []
    for ll in list(library.keys()):
        if intersect:
            preds.append(correlation.loc[:, library[ll]].loc[ugenes,:].mean(axis=1))
        else:
            preds.append(correlation.loc[:, library[ll]].mean(axis=1))
    predictions = pd.concat(preds, axis=1)
    predictions.columns = list(library.keys())
    predictions = pd.DataFrame(predictions.fillna(0), dtype=np.float32)
    predictions = predictions.reset_index(drop=True)
    predictions.columns = predictions.columns.astype(str)
    predictions.to_feather(outFolder+"/prediction_"+str(i)+".f")

def loadCorrelation(correlationFolder: str, suffix: str):
    cct = pd.read_feather(correlationFolder+"/correlation_"+str(suffix)+".f")
    idx = [x.upper() for x in cct.columns]
    cct.index = idx
    cct.columns = idx
    xx = cct.to_numpy()
    np.fill_diagonal(xx, float('nan'))
    return(cct)

def loadPrediction(predictionFolder: str, i: int):
    return pd.read_feather(predictionFolder+"/prediction_"+str(i)+".f")

def loadPredictions(predictionFolder: str, verbose: bool=False): 
    predictions = []
    prediction_files = os.listdir(predictionFolder)
    lk = list(range(0, len(prediction_files)-1))
    lk.append("global")
    for i in lk:
        prediction = pd.DataFrame(pd.read_feather(predictionFolder+"/prediction_"+str(i)+".f"), dtype=np.float32)
        prediction = prediction.fillna(0)
        predictions.append(prediction)
        if verbose:
            print("prediction_"+str(i)+".f")
    return(predictions)

def loadPredictionsRange(predictionFolder: str, rangeFrom: int, rangeTo: int, verbose: bool=False): 
    predictions = []
    prediction_files = os.listdir(predictionFolder)
    lk = list(range(0, len(prediction_files)-1))
    lk.append("global")
    for i in lk:
        prediction = pd.DataFrame(pd.read_feather(predictionFolder+"/prediction_"+str(i)+".f"), dtype=np.float32)
        prediction = prediction.fillna(0)
        predictions.append(prediction.iloc[:, rangeFrom:rangeTo].copy())
        prediction = 0
        if verbose:
            print("prediction_"+str(i)+".f")
    return(predictions)
