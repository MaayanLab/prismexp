import pandas as pd
import numpy as np
from typing import List
import itertools
import urllib.request
import json
import os
import re
import math
import feather
import qnorm

def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    input: dataframe with numerical columns
    output: dataframe with quantile normalized values
    """
    df_sorted = pd.DataFrame(np.sort(df.values, axis=0), 
                             index=df.index, 
                             columns=df.columns, dtype=np.float32)
    print("peak")
    df_mean = df_sorted.mean(axis=1)
    df_sorted = 0
    df_mean.index = np.arange(1, len(df_mean) + 1)
    df_qn = df.rank(method="min").stack().astype(int).map(df_mean).unstack()
    return(df_qn)

def readGMT(gmtFile: str, backgroundGenes: List[str]=[], verbose=False) -> List:
    file = open(gmtFile, 'r')
    lines = file.readlines()
    library = {}
    backgroundSet = {}
    if len(backgroundGenes) > 1:
        backgroundGenes = [x.upper() for x in backgroundGenes]
        backgroundSet = set(backgroundGenes)
    for line in lines:
        sp = line.strip().upper().split("\t")
        sp2 = [re.sub(",.*", "",value) for value in sp[2:]]
        if len(backgroundGenes) > 2:
            geneset = list(set(sp2).intersection(backgroundSet))
            if len(geneset) > 0:
                library[sp[0]] = geneset
        else:
            library[sp[0]] = sp2
    ugenes = list(set(list(itertools.chain.from_iterable(library.values()))))
    ugenes.sort()
    rev_library = {}
    for ug in ugenes:
        rev_library[ug] = []
    for se in library.keys():
        for ge in library[se]:
            rev_library[ge].append(se)
    if verbose:
        print("Library loaded. Library contains "+str(len(library))+" gene sets. "+str(len(ugenes))+" unique genes found.")
    return [library, rev_library, ugenes]

def loadJSON(url):
    req = urllib.request.Request(url)
    r = urllib.request.urlopen(req).read()
    return(json.loads(r.decode('utf-8')))

def getConfig():
    config_url = os.path.join(
        os.path.dirname(__file__),
        'data/config.json')
    with open(config_url) as json_file:
        data = json.load(json_file)
    return(data)

def getDataPath() -> str:
    path = os.path.join(
        os.path.dirname(__file__),
        'data/'
    )
    return(path)

def help():
    help = os.path.join(
        os.path.dirname(__file__),
        'data/help.txt'
    )
    with open(help, 'r') as f:
        data = f.read()
        print(data)

def normalize(exp: pd.DataFrame, stepSize: int=2000, transpose: bool=False) -> pd.DataFrame:
    if transpose: exp = exp.transpose()
    exp = pd.DataFrame(np.log2(exp+1))
    exp = qnorm.quantile_normalize(exp)
    return(exp)

def loadCorrelation(correlationFolder: str, suffix: int):
    cc = pd.DataFrame(pd.read_feather(correlationFolder+"/correlation_"+str(suffix)+".f").set_index("index"), dtype=np.float32)
    return(cc)

def loadPrediction(predictionFolder: str, i: int):
    return pd.DataFrame(pd.read_feather(predictionFolder+"/prediction_"+str(i)+".f").set_index("index"), dtype=np.float32)
