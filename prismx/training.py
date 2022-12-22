import pandas as pd
import numpy as np
import os
import math
import random
import pickle
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
from progress.bar import Bar
import tqdm

from prismx.utils import read_gmt, load_correlation, load_feature
from prismx.feature import load_features

def create_training_data(workdir: str, gmt_file: str, false_sample_count: int=50000, verbose: bool=True) -> List:
    correlation_files = os.listdir(workdir+"/correlation")
    correlation = load_correlation(workdir, 0)
    background_genes = list(correlation.columns)
    library, rev_library, ugenes = read_gmt(gmt_file, background_genes)
    df_true = []
    lk = list(range(0, len(correlation_files)-1))
    lk.append("global")
    
    for i in tqdm.tqdm(lk, desc="Build True Examples", disable=(not verbose)):
        feature = load_feature(workdir, i)
        features = []
        keys = list(feature.columns)
        setname = []
        genename = []
        for se in keys:
            vals = library[se]
            for val in vals:
                setname.append(val)
                genename.append(se)
                features.append(feature.loc[val, se])
        df_true.append(features)
    #df_true = pd.concat(df_true, axis=1)
    df_true = pd.DataFrame(np.array(df_true).T)
        
    df_true2 = pd.concat([pd.DataFrame(genename), pd.DataFrame(setname),df_true, pd.DataFrame(np.ones(len(setname)))], axis=1)
    samp_set = []
    samp_gene = []
    npw = np.array(df_true2.iloc[:, 0])
    false_gene_count = math.ceil(false_sample_count/len(background_genes))
    for i in background_genes:
        rkey = random.sample(keys,1)[0]
        ww = np.where(npw == rkey)[0]
        for j in range(0, false_gene_count):
            rgene = random.sample(background_genes,1)[0]
            if rgene not in df_true2.iloc[ww, 1]:
                samp_set.append(rkey)
                samp_gene.append(rgene)
    df_false = []
    
    for i in tqdm.tqdm(lk, desc="Build False Examples", disable=(not verbose)):
        feature = load_feature(workdir, i)
        features = []
        setname = []
        genename = []
        for k in range(0,len(samp_set)):
            se = samp_set[k]
            val = samp_gene[k]
            setname.append(se)
            genename.append(val)
            features.append(feature.loc[val, se])
        df_false.append(features)
    df_false = pd.DataFrame(np.array(df_false).T)
    df_false2 = pd.concat([pd.DataFrame(setname), pd.DataFrame(genename),df_false,pd.DataFrame(np.zeros(len(setname)))], axis=1)
    return([df_true2, df_false2.iloc[random.sample(range(0, df_false2.shape[0]), false_sample_count), :]])

def balance_data(df_true: pd.DataFrame, df_false: pd.DataFrame, true_count: int, false_count: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    true_count = min(true_count, df_true.shape[0])
    false_count = min(false_count, df_false.shape[0])
    rtrue = random.sample(list(range(0, df_true.shape[0])), true_count)
    rtrue.sort()
    rfalse = random.sample(list(range(0, df_false.shape[0])), false_count)
    rfalse.sort()
    df_combined = pd.concat([df_true.iloc[rtrue,:], df_false.iloc[rfalse,:]])
    df_combined = df_combined.reset_index()
    X = df_combined.iloc[:,3:(df_combined.shape[1]-1)]
    y = df_combined.iloc[:,df_combined.shape[1]-1]
    return(X, y)

def train(workdir: str, gmt_file: str, training_size: int=200000, test_train_split: float=0.1, sample_positive: int=20000, sample_negative: int=80000, random_state: int=42, verbose: bool=False):
    df_true, df_false = create_training_data(workdir, gmt_file, training_size, verbose=verbose)
    X, y = balance_data(df_true, df_false, sample_positive, sample_negative)
    true_count = np.sum(y)
    false_count = len(y)-true_count
    if verbose: print("positive samples: "+str(round(true_count))+"\nnegative samples: "+str(round(false_count)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_train_split, random_state=random_state)
    model = LGBMRegressor(seed=42)
    model.fit(X_train, y_train)
    pickle.dump(model, open(workdir+"/model.pkl", 'wb'))
    return(model)
