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
from sklearn.ensemble import RandomForestClassifier

from prismx.utils import readGMT, loadCorrelation, loadPrediction
from prismx.prediction import correlationScores, loadPredictionsRange

def predictGMT(model, gmtFile: str, correlationFolder: str, predictionFolder: str, outFolder: str, predictionName: str, stepSize: int=500, intersect: bool=False, verbose: bool=False):
    os.makedirs(outFolder, exist_ok=True)
    os.makedirs(predictionFolder, exist_ok=True)
    correlationScores(gmtFile, correlationFolder, predictionFolder, intersect=intersect, verbose=verbose)
    prismxPredictions(model, predictionFolder, predictionName, outFolder, stepSize, verbose=verbose)

def prismxPredictions(model: str, predictionFolder: str, predictionName: str, outFolder: str, stepSize: int=500, verbose: bool=False):
    predictionSize = loadPrediction(predictionFolder, 0).shape[1]
    prism = pd.DataFrame()
    stepNumber = math.ceil(predictionSize/stepSize)
    if verbose: bar = Bar('Processing predictions', max=stepNumber)
    for i in range(0, stepNumber):
        rfrom = i*stepSize
        rto = min((i+1)*stepSize, predictionSize)
        predictions = loadPredictionsRange(predictionFolder, rfrom, rto)
        prism = makePredictionsRange(model, prism, predictions)
        predictions = 0
        if verbose: bar.next()
    if verbose: bar.finish()
    prism.reset_index().to_feather(outFolder+"/"+predictionName+".f")

def makePredictionsRange(model: str, prism: pd.DataFrame, predictions: List[pd.DataFrame], verbose: bool=False) -> pd.DataFrame:
    model = pickle.load(open(model, 'rb'))
    predList = []
    for i in range(0, predictions[0].shape[1]):
        start = time.time()
        df = pd.DataFrame()
        k = 0
        for pp in predictions:
            df[k] = pp.iloc[:,i]
            k = k + 1
        if verbose:
            print(str(i) + " - " + str(round(time.time()-start)))
        df.fillna(0, inplace=True)
        predList.append(model.predict_proba(df)[:,1])
    prismTemp = pd.DataFrame(predList).transpose()
    prismTemp.columns = predictions[0].columns
    prismTemp.index = predictions[0].index
    if prism.shape[1] == 0:
        prism = prismTemp
    else:
        print(prism.shape)
        print(prismTemp.shape)
        prism = pd.concat((prism, prismTemp), axis=1)
        print(prism.shape)
    return(prism)
