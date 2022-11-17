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

def read_gmt(gmt_file: str, background_genes: List[str]=[], verbose=False) -> List:
    file = open(gmt_file, 'r')
    lines = file.readlines()
    library = {}
    background_set = {}
    if len(background_genes) > 1:
        background_genes = [x.upper() for x in background_genes]
        background_set = set(background_genes)
    for line in lines:
        sp = line.strip().upper().split("\t")
        sp2 = [re.sub(",.*", "",value) for value in sp[2:]]
        sp2 = [x for x in sp2 if x] 
        if len(background_genes) > 2:
            geneset = list(set(sp2).intersection(background_set))
            if len(geneset) > 0:
                library[sp[0]] = geneset
        else:
            if len(sp2) > 0:
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

def load_json(url):
    req = urllib.request.Request(url)
    r = urllib.request.urlopen(req).read()
    return(json.loads(r.decode('utf-8')))

def get_config():
    config_url = os.path.join(
        os.path.dirname(__file__),
        'data/config.json')
    with open(config_url) as json_file:
        data = json.load(json_file)
    return(data)

def get_data_path() -> str:
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

def normalize(exp: pd.DataFrame, transpose: bool=False) -> pd.DataFrame:
    if transpose: exp = exp.transpose()
    exp = pd.DataFrame(np.log2(exp+1))
    exp_qnorm = qnorm.quantile_normalize(np.array(exp, dtype=np.float32))
    exp = pd.DataFrame(exp_qnorm, index=exp.index, columns=exp.columns)
    return(exp)

def load_correlation(workdir: str, suffix: int) -> pd.DataFrame:
    cc = pd.DataFrame(pd.read_feather(workdir+"/correlation/correlation_"+str(suffix)+".f").set_index("index"), dtype=np.float16)
    return(cc)

def load_feature(workdir: str, i: int) -> pd.DataFrame:
    return pd.DataFrame(pd.read_feather(workdir+"/features/features_"+str(i)+".f").set_index("index"), dtype=np.float32)
