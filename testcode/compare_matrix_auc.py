import pandas as pd
import feather
import numpy as np
import os
import math
import random
import pickle
import time

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import prismx as px
from prismx.validation import calculateGeneAUC, calculateSetAUC
from prismx.utils import readGMT

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2]
    return lst3 


correlationFolder = "correlation_folder_300_q"
predictionFolder = "prediction_folder_300_q"
outfolder = "prismxresult_q"



aucs = pd.read_feather("testdata/gobp_all_avg.f")
aucs = aucs.set_index("index")
libs = px.listLibraries()
gmtFile = px.loadLibrary(libs[111], overwrite=True)
outname = libs[111]
from prismx.validation import calculateSetAUC,calculateGeneAUC, benchmarkGMT, getGenes
geneAUC2, setAUC2 = benchmarkGMT(gmtFile, correlationFolder, predictionFolder, outfolder+"/"+outname+".f", intersect=True, verbose=True)

pd.write_feather()

gop = pd.read_feather(outfolder+"/GO_Biological_Process_2018.f")
gop = gop.set_index("index")
gop.index = [x.decode("UTF-8") for x in gop.index]

[libr, rev_libr, ugenes] = readGMT(gmtFile, backgroundGenes=gop.index)

pred = pd.read_feather(predictionFolder+"/prediction_0.f")
pred = pred.set_index("index")
pred.index = [x.decode("UTF-8") for x in pred.index]



aucs = pd.read_feather("testdata/gobp_all_avg.f")
aucs = aucs.set_index("index")

clustn = 300

libs = px.listLibraries()
gmtFile = px.loadLibrary(libs[111], overwrite=True)

outname = libs[111]

geneAUC, setAUC = px.benchmarkGMTfast(gmtFile, correlationFolder, predictionFolder, outfolder+"/"+outname+".f", intersect=True, verbose=True)

inter = intersection(setAUC.index, aucs.index)
inter.sort()

aucs = aucs.loc[inter,:]
setAUC = setAUC.loc[inter,:]

aucs["prismx"] = setAUC.loc[:,"prismx"]
auci = aucs.reset_index()
auci.to_feather("testdata/matrix_auc.f")
