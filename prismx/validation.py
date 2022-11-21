from sklearn.metrics import roc_auc_score, roc_curve, auc
import pandas as pd
import numpy as np
from typing import Dict, List
from progress.bar import Bar
import tqdm
import os
import pickle
from prismx.utils import read_gmt, load_correlation, load_feature
from prismx.loaddata import get_genes

def calculate_set_auc(prediction: pd.DataFrame, library: Dict, min_lib_size: int=1) -> pd.DataFrame:
    aucs = []
    setnames = []
    gidx = prediction.index
    for se in library:
        if len(library[se]) >= min_lib_size:
            lenc = library[se]
            gold = [i in lenc for i in gidx]
            fpr, tpr, _ = roc_curve(list(gold), list(prediction.loc[:,se]))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            setnames.append(se)
    aucs = pd.DataFrame(aucs, index=setnames)
    return(aucs)

def calculate_gene_auc(prediction: pd.DataFrame, rev_library: Dict, min_lib_size: int=1) -> List[float]:
    aucs = []
    genes = []
    gidx = prediction.index
    for se in rev_library:
        gold = [i in rev_library[se] for i in prediction.columns]
        if len(rev_library[se]) >= min_lib_size and se in gidx:
            fpr, tpr, _ = roc_curve(list(gold), list(prediction.loc[se,:]))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            genes.append(se)
    aucs = pd.DataFrame(aucs, index=genes)
    return(aucs)

def benchmark_gmt(gmt_file: str, workdir: str, prediction_file: str, intersect: bool=False, verbose=False):
    genes = get_genes(workdir)
    library, rev_library, unique_genes = read_gmt(gmt_file, genes, verbose=verbose)
    if intersect:
        ugenes = list(set(sum(library.values(), [])))
        genes = list(set(ugenes) & set(genes))
    prediction_files = os.listdir(workdir+"/features")
    lk = list(range(0, len(prediction_files)-1))
    lk.append("global")
    geneAUC = []
    setAUC = []
    for i in tqdm.tqdm(lk, desc="AUC calculation", disable=(not verbose)):
        prediction = load_feature(workdir, i).loc[genes,:]
        prediction.index = prediction.index
        prediction = prediction.loc[genes,:]
        geneAUC.append(calculate_gene_auc(prediction, rev_library))
        setAUC.append(calculate_set_auc(prediction, library)[0])
    
    geneAUC = pd.DataFrame(np.array(geneAUC).T, index=genes)
    prediction = pd.read_feather(prediction_file).set_index("index").loc[genes,:]
    geneAUC["prismx"] = calculate_gene_auc(prediction, rev_library)
    geneAUC.index = unique_genes
    setAUC["prismx"] = calculate_set_auc(prediction, library)[0]
    return([geneAUC, setAUC])

def benchmarkGMTfast(gmt_file: str, correlationFolder: str, predictionFolder: str, prismxPrediction: str, minLibSize: int=1, intersect: bool=False, verbose=False):
    genes = get_genes(correlationFolder)
    library, rev_library, unique_genes = read_gmt(gmt_file, genes, verbose=verbose)
    if intersect:
        ugenes = list(set(sum(library.values(), [])))
        genes = list(set(ugenes) & set(genes))
    unique_genes = [x.encode('utf-8') for x in unique_genes]
    prediction_files = os.listdir(predictionFolder)
    geneAUC = pd.DataFrame()
    setAUC = pd.DataFrame()
    prediction = load_feature(predictionFolder, "global").loc[unique_genes,:]
    geneAUC["global"] = calculate_gene_auc(prediction, rev_library)
    setAUC["global"] = calculate_set_auc(prediction, library)[0]
    prediction = pd.read_feather(prismxPrediction).set_index("index").loc[unique_genes,:]
    geneAUC["prismx"] = calculate_gene_auc(prediction, rev_library)
    geneAUC.index = unique_genes
    setAUC["prismx"] = calculate_set_auc(prediction, library)[0]
    return([geneAUC, setAUC])

def benchmark_gmt_fast(gmt_file: str, workdir: str, prediction_file: str, intersect: bool=False, verbose=False):
    genes = [x.decode("UTF-8") for x in get_genes(workdir)]
    library, rev_library, unique_genes = read_gmt(gmt_file, genes, verbose=verbose)
    if intersect:
        ugenes = list(set(sum(library.values(), [])))
        genes = list(set(ugenes) & set(genes))
    prediction_files = os.listdir(workdir+"/features")
    lk = list(range(0, len(prediction_files)-1))
    lk.append("global")
    geneAUC = pd.DataFrame()
    setAUC = pd.DataFrame()
    genes = [x.encode("UTF-8") for x in genes]
    prediction = pd.read_feather(prediction_file).set_index("index").loc[genes,:]
    geneAUC["prismx"] = calculate_gene_auc(prediction, rev_library)
    geneAUC.index = unique_genes
    setAUC["prismx"] = calculate_set_auc(prediction, library)[0]
    return([geneAUC, setAUC])
