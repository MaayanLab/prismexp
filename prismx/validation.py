from sklearn.metrics import roc_auc_score, roc_curve, auc
import pandas as pd
from typing import Dict, List
from progress.bar import Bar
import os
import pickle
from prismx.utils import readGMT, loadCorrelation, loadPrediction
from prismx.loaddata import getGenes

def calculateSetAUC(prediction: pd.DataFrame, library: Dict, minLibSize: int=1) -> List[float]:
    aucs = []
    setnames = []
    idx = prediction.index
    for se in library:
        gold = [i in library[se] for i in idx]
        if len(library[se]) >= minLibSize:
            fpr, tpr, _ = roc_curve(list(gold), list(prediction.loc[:,se]))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            setnames.append(se)
    aucs = pd.DataFrame(aucs, index=setnames)
    return(aucs)

def calculateGeneAUC(prediction: pd.DataFrame, rev_library: Dict, minLibSize: int=1) -> List[float]:
    aucs = []
    idx = prediction.index
    for se in rev_library:
        gold = [i in rev_library[se] for i in prediction.columns]
        if len(rev_library[se]) >= minLibSize and se in idx:
            fpr, tpr, _ = roc_curve(list(gold), list(prediction.loc[se,:]))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
    return(aucs)

def benchmarkGMT(gmtFile: str, correlationFolder: str, predictionFolder: str, prismxPrediction: str, minLibSize: int=1, intersect: bool=False, verbose=False):
    genes = getGenes(correlationFolder)
    library, revLibrary, uniqueGenes = readGMT(gmtFile, genes, verbose=verbose)
    if intersect:
        ugenes = list(set(sum(library.values(), [])))
        genes = list(set(ugenes) & set(genes))
    prediction_files = os.listdir(predictionFolder)
    lk = list(range(0, len(prediction_files)-1))
    lk.append("global")
    geneAUC = pd.DataFrame()
    setAUC = pd.DataFrame()
    if verbose: bar = Bar('AUC calculation', max=len(lk))
    for i in lk:
        prediction = loadPrediction(predictionFolder, i)
        geneAUC[i] = calculateGeneAUC(prediction, revLibrary)
        setAUC[i] = calculateSetAUC(prediction, library)[0]
        if verbose: bar.next()
    if verbose: bar.finish()
    prediction = pd.read_feather(prismxPrediction).set_index("index")
    geneAUC["prismx"] = calculateGeneAUC(prediction, revLibrary)
    geneAUC.index = uniqueGenes
    setAUC["prismx"] = calculateSetAUC(prediction, library)[0]
    return([geneAUC, setAUC])

def benchmarkGMTfast(gmtFile: str, correlationFolder: str, predictionFolder: str, prismxPrediction: str, minLibSize: int=1, intersect: bool=False, verbose=False):
    genes = getGenes(correlationFolder)
    library, revLibrary, uniqueGenes = readGMT(gmtFile, genes, verbose=verbose)
    if intersect:
        ugenes = list(set(sum(library.values(), [])))
        genes = list(set(ugenes) & set(genes))
    prediction_files = os.listdir(predictionFolder)
    geneAUC = pd.DataFrame()
    setAUC = pd.DataFrame()
    prediction = loadPrediction(predictionFolder, "global")
    geneAUC["global"] = calculateGeneAUC(prediction, revLibrary)
    setAUC["global"] = calculateSetAUC(prediction, library)[0]
    prediction = pd.read_feather(prismxPrediction).set_index("index")
    prediction.index = genes
    geneAUC["prismx"] = calculateGeneAUC(prediction, revLibrary)
    geneAUC.index = uniqueGenes
    setAUC["prismx"] = calculateSetAUC(prediction, library)[0]
    return([geneAUC, setAUC])
