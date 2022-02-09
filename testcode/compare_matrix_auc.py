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
from prism.utils import read_gmt

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2]
    return lst3 


correlationFolder = "correlation_folder_300_q"
predictionFolder = "prediction_folder_300_q"
outfolder = "prismxresult_q"



aucs = pd.read_feather("test_data/gobp_all_avg.f")
aucs = aucs.set_index("index")
libs = px.list_libraries()
gmt_file = px.load_library(libs[111], overwrite=True)
outname = libs[111]
geneAUC, setAUC = px.benchmarkGMTfast(gmt_file, correlationFolder, predictionFolder, outfolder+"/"+outname+".f", intersect=True, verbose=True)


pd.write_feather()

gop = pd.read_feather(outfolder+"/GO_Biological_Process_2018.f")
gop = gop.set_index("index")
gop.index = [x.decode("UTF-8") for x in gop.index]

[libr, rev_libr, ugenes] = readGMT(gmtFile, backgroundGenes=gop.index)

pred = pd.read_feather(predictionFolder+"/prediction_0.f")
pred = pred.set_index("index")
pred.index = [x.decode("UTF-8") for x in pred.index]



aucs = pd.read_feather("test_data/gobp_all_avg.f")
aucs = aucs.set_index("index")

clustn = 300

libs = px.list_libraries()
gmt_file = px.load_library(libs[111], overwrite=True)

outname = libs[111]

geneAUC, setAUC = px.benchmarkGMTfast(gmt_file, correlationFolder, predictionFolder, outfolder+"/"+outname+".f", intersect=True, verbose=True)

inter = intersection(setAUC.index, aucs.index)
inter.sort()

aucs = aucs.loc[inter,:]
setAUC = setAUC.loc[inter,:]

aucs["prismx"] = setAUC.loc[:,"prismx"]
auci = aucs.reset_index()
auci.to_feather("test_data/matrix_auc.f")
