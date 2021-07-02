import pandas as pd
import numpy as np
import os
import math
import random
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from progress.bar import Bar

from prismx.utils import readGMT, loadCorrelation, loadPrediction
from prismx.prediction import loadPredictions

def createTrainingData(predictionFolder: str, correlationFolder: str, gmtFile: str, falseSampleCount: int=50000) -> List:
    correlation_files = os.listdir(correlationFolder)
    correlation = loadCorrelation(correlationFolder, 0)
    backgroundGenes = list(correlation.columns)
    library, rev_library, ugenes = readGMT(gmtFile, backgroundGenes)
    df_true = pd.DataFrame()
    lk = list(range(0, len(correlation_files)-1))
    lk.append("global")
    bar = Bar('Retrieve training data', max=2*len(lk))
    for i in lk:
        predictions = loadPrediction(predictionFolder, i)
        pred = []
        keys = list(predictions.columns)
        setname = []
        genename = []
        for se in keys:
            vals = library[se]
            for val in vals:
                setname.append(val)
                genename.append(se)
                pred.append(predictions.loc[val.encode('UTF-8'), se])
        df_true.loc[:,i] = pred
        bar.next()
    df_true2 = pd.concat([pd.DataFrame(genename), pd.DataFrame(setname),df_true, pd.DataFrame(np.ones(len(setname)))], axis=1)
    samp_set = []
    samp_gene = []
    npw = np.array(df_true2.iloc[:, 0])
    falseGeneCount = math.ceil(falseSampleCount/len(backgroundGenes))
    for i in backgroundGenes:
        rkey = random.sample(keys,1)[0]
        ww = np.where(npw == rkey)[0]
        for j in range(0, falseGeneCount):
            rgene = random.sample(backgroundGenes,1)[0]
            if rgene not in df_true2.iloc[ww, 1]:
                samp_set.append(rkey)
                samp_gene.append(rgene)
    df_false = pd.DataFrame()
    Bar('Retrieve false samples ', max=len(lk))
    for i in lk:
        predictions = loadPrediction(predictionFolder, i)
        pred = []
        setname = []
        genename = []
        for k in range(0,len(samp_set)):
            se = samp_set[k]
            val = samp_gene[k]
            setname.append(se)
            genename.append(val)
            pred.append(predictions.loc[val.encode('UTF-8'), se])
        df_false.loc[:,i] = pred
        bar.next()
    df_false2 = pd.concat([pd.DataFrame(setname), pd.DataFrame(genename),df_false,pd.DataFrame(np.zeros(len(setname)))], axis=1)
    bar.finish()
    return([df_true2, df_false2.iloc[random.sample(range(0, df_false2.shape[0]), falseSampleCount), :]])

def balanceData(df_true: pd.DataFrame, df_false: pd.DataFrame, trueCount: int, falseCount: int) -> str:
    trueCount = min(trueCount, df_true.shape[0])
    falseCount = min(falseCount, df_false.shape[0])
    rtrue = random.sample(list(range(0, df_true.shape[0])), trueCount)
    rtrue.sort()
    rfalse = random.sample(list(range(0, df_false.shape[0])), falseCount)
    rfalse.sort()
    df_combined = pd.concat([df_true.iloc[rtrue,:], df_false.iloc[rfalse,:]])
    df_combined = df_combined.reset_index()
    X = df_combined.iloc[:,3:(df_combined.shape[1]-1)]
    y = df_combined.iloc[:,df_combined.shape[1]-1]
    return(X, y)

def trainModel(predictionFolder: str, correlationFolder: str, gmtFile: str, trainingSize: int=200000, testTrainSplit: float=0.1, samplePositive: int=20000, sampleNegative: int=80000, randomState: int=42, verbose: bool=False):
    df_true, df_false = createTrainingData(predictionFolder, correlationFolder, gmtFile, trainingSize)
    X, y = balanceData(df_true, df_false, samplePositive, sampleNegative)
    trueCount = np.sum(y)
    falseCount = len(y)-trueCount
    if verbose: print("positive samples: "+str(round(trueCount))+"\nnegative samples: "+str(round(falseCount)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testTrainSplit, random_state=randomState)
    model = RandomForestClassifier(random_state=randomState)
    model.fit(X_train, y_train)
    return(model)
