
import pandas as pd
import numpy as np
import os
import math
import random
import pickle
import time
import plotly.express as pl

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

import prismx as px

from prismx.utils import readGMT, loadCorrelation, loadPrediction
from prismx.prediction import correlationScores, loadPredictionsRange
from sklearn.manifold import TSNE


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2]
    return lst3 


gene_auc = pd.read_csv("testdata/gene_auc.tsv", sep="\t", index_col=0)
set_auc = pd.read_csv("testdata/set_auc.tsv", sep="\t", index_col=0)

diff = gene_auc.iloc[:,5] - gene_auc.iloc[:,0]
diff.sort_values(0,ascending=False).iloc[0:20]


diff = set_auc.iloc[:,5] - set_auc.iloc[:,0]
diff.sort_values(0,ascending=False).iloc[0:20]

nx = "GO_Biological_Process_2018"

set_auc.loc[nx,:]
gene_auc.loc[nx,:]


p1 = pd.read_feather("prediction_folder_300_umap/prediction_0.f").set_index("index")


correlationFolder = "correlation_folder_300"
predictionFolder = "prediction_folder_300_umap"
outfolder = "prismxresult"

clustn = 300

libs = px.listLibraries()
gmtFile = px.loadLibrary(libs[111], overwrite=True)

outname = libs[111]
#px.predictGMT("gobp_model_"+str(clustn)+".pkl", gmtFile, correlationFolder, predictionFolder, outfolder, libs[111], stepSize=200, intersect=True, verbose=True)

gop = pd.read_feather("prismxresult/GO_Biological_Process_2018.f")
gop = gop.set_index("index")
geneAUC, setAUC = px.benchmarkGMTfast(gmtFile, correlationFolder, predictionFolder, outfolder+"/"+outname+".f", intersect=True, verbose=True)

diff_gene = geneAUC.iloc[:,1]-geneAUC.iloc[:,0]
diff_set = setAUC.iloc[:,1]-setAUC.iloc[:,0]
sort_d = diff_set.sort_values(0)


plt.scatter(setAUC.iloc[:,0], setAUC.iloc[:,1], c="black")
plt.plot([0,1], [0,1], linestyle='--')
ax = plt.gca()

ax.annotate("text", (0.8, 0.2), xytext=(0.8+0.05, 0.2+0.3), 
    arrowprops = dict(  arrowstyle="->",
                        connectionstyle="angle3,angleA=0,angleB=-90"))

plt.savefig("figures/tempimp.pdf")
plt.close()

dict, rdict, ugenes = px.readGMT(gmtFile)
dict[sort_d.index[-1]]

i = 1
d1 = sort_d.index[-i]
dgenes = [x.encode("UTF-8") for x in dict[d1]]

sort_d.loc[d1,]


pk = pd.read_feather("correlation_folder_300/correlation_global.f")
pk = pk.set_index("index")

p1 = pd.read_feather("prediction_folder_300_umap/prediction_0.f").set_index("index")


mm = pk.loc[:, dict[d1]].mean(axis=1)
gold = [i in dgenes for i in pk.index]
fpr, tpr, _ = roc_curve(list(gold), mm)
roc_auc = auc(fpr, tpr)

ll = list(range(20))
ll.append("global")

aucs = list()
improved = pd.DataFrame()
setname = list()
for i in ll:
    print(i)
    pk = pd.read_feather("correlation_folder_300/correlation_"+str(i)+".f")
    pk = pk.set_index("index")
    aucs = list()
    setname = list()
    for j in range(1,101):
        d1 = sort_d.index[-j]
        setname.append(d1)
        #print(d1+"\t"+str(sort_d[-j])+"\t"+str(len(intersection(dict[d1],pk.columns))))
        mm = pk.loc[:, dict[d1]].mean(axis=1).fillna(0)
        gold = [x in dict[d1] for x in pk.columns]
        fpr, tpr, _ = roc_curve(list(gold), mm)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    improved[i] = aucs

improved.index = setname


ll = list(range(300))
ll.append("global")

aucs = list()
allavg = pd.DataFrame()
setname = list()
for i in ll:
    print(i)
    pk = pd.read_feather("correlation_folder_300/correlation_"+str(i)+".f")
    pk = pk.set_index("index")
    aucs = list()
    setname = list()
    for j in range(sort_d.shape[0]):
        d1 = sort_d.index[j]
        setname.append(d1)
        #print(d1+"\t"+str(sort_d[-j])+"\t"+str(len(intersection(dict[d1],pk.columns))))
        mm = pk.loc[:, dict[d1]].mean(axis=1).fillna(0)
        gold = [x in dict[d1] for x in pk.columns]
        fpr, tpr, _ = roc_curve(list(gold), mm)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    allavg[i] = aucs

allavg.index = setname



kk = 2


l = 2
plt.close()
plt.bar(range(worse.shape[1]), list(worse.iloc[l,:]))
plt.bar(worse.shape[1]-1, list(worse.iloc[l,:])[-1], color="red")
plt.savefig("figures/tempimp.pdf")
plt.close()



l = 99
plt.close()
plt.bar(range(improved.shape[1]), list(improved.iloc[l,:]))
plt.bar(improved.shape[1]-1, list(improved.iloc[l,:])[-1], color="red")
plt.savefig("figures/tempimp.pdf")
plt.close()




aucs = list()
worse = pd.DataFrame()


for i in ll:
    print(i)
    pk = pd.read_feather("correlation_folder_300/correlation_"+str(i)+".f")
    pk = pk.set_index("index")
    aucs = list()
    setname = list()
    for j in range(100):
        d1 = sort_d.index[j]
        setname.append(d1)
        #print(d1+"\t"+str(sort_d[j]))
        mm = pk.loc[:, dict[d1]].mean(axis=1).fillna(0)
        gold = [x in dict[d1] for x in pk.columns]
        fpr, tpr, _ = roc_curve(list(gold), mm)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    worse[i] = aucs

worse.index = setname

