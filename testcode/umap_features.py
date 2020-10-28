import pandas as pd
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

from prismx.utils import readGMT, loadCorrelation, loadPrediction
from prismx.prediction import correlationScores, loadPredictionsRange
from sklearn.manifold import TSNE


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
diff_set.sort_values(0)

dic, rdic, ugenes = px.readGMT(gmtFile, backgroundGenes=diff_gene.index)

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2]
    return lst3 

kk = intersection(list(dic.keys()), diff_set.index)
ll1 = []
ll2 = []
for i in range(len(kk)):
    #print(kk[i]+" - "+str(diff_set.loc[kk[i]])+" - "+str(len(dic[kk[i]])))
    ll1.append(diff_set.loc[kk[i]])
    ll2.append(len(dic[kk[i]]))

c1 = np.corrcoef(ll1,ll2)[0][1]

plt.scatter(ll1, ll2, s=0.7, alpha=0.5, color="black")
plt.xlabel("prediction improvement", fontsize=15)
plt.ylabel("gene set size", fontsize=15)
plt.text(-0.38, 1200, "cor: "+str(round(c1,ndigits=4)), fontsize=15)
plt.savefig("figures/set_size_improvement.pdf")
plt.close()


kk = intersection(list(rdic.keys()), diff_gene.index)
ll1 = []
ll2 = []
for i in range(len(kk)):
    #print(kk[i]+" - "+str(diff_set.loc[kk[i]])+" - "+str(len(dic[kk[i]])))
    ll1.append(diff_gene.loc[kk[i]])
    ll2.append(len(rdic[kk[i]]))

c2 = np.corrcoef(ll1,ll2)[0][1]

plt.scatter(ll1, ll2, s=0.7, alpha=0.5, color="black")
plt.xlabel("prediction improvement", fontsize=15)
plt.ylabel("gene annotations", fontsize=15)
plt.text(-0.7, 230, "cor: "+str(round(c2,ndigits=4)), fontsize=15)
plt.savefig("figures/gene_size_improvement.pdf")
plt.close()


df = pd.DataFrame()
k = 0
print(i)

kk = intersection(list(dic.keys()), diff_set.index)
rr = random.sample(range(len(kk)), 200)
rr.sort()

predictionFolder = "prediction_folder_300_umap"
pp = []
pcount = list(range(300))
pcount.append("global")
for i in pcount:
    print(i)
    pp.append(loadPrediction(predictionFolder, i).iloc[:, rr])


ugene2 = [x.encode("UTF-8") for x in ugenes]
true = []
false = []
for i in pp[0].columns:
    genes = [x.encode("UTF-8") for x in dic[i]]
    ff = random.sample(list(set(ugene2).difference(set(genes))), 50)
    mat = pd.DataFrame()
    kk = 0
    for p in pp:
        mat[str(pcount[kk])] = p.loc[genes,i]
        kk = kk+1
    mat["prismx"] = gop.loc[genes,i]
    true.append(mat)
    mat = pd.DataFrame()
    kk = 0
    for p in pp:
        mat[str(pcount[kk])] = p.loc[ff,i]
        kk = kk+1
    mat["prismx"] = gop.loc[ff,i]
    false.append(mat)

true_all = pd.concat(true)
false_all = pd.concat(false)

samples_all = pd.concat([true_all, false_all])

tt = TSNE(n_components=2).fit_transform(samples_all.iloc[:,0:301])


plt.figure(figsize=(7,6))
plt.scatter(tt[:true_all.shape[0],0], tt[:true_all.shape[0],1], s=0.7, alpha=0.5, color="red")
plt.scatter(tt[true_all.shape[0]:,0], tt[true_all.shape[0]:,1], s=0.7, alpha=0.5, color="black")
plt.xlabel("T1", fontsize=15)
plt.ylabel("T2", fontsize=15)
colors = {'true samples':'red', 'false samples':'black'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.savefig("figures/tsne_features.pdf")
plt.close()

sc = samples_all.iloc[:,301]
sc = sc.fillna(0)

plt.scatter(tt[:,0], tt[:,1], s=0.7, alpha=0.5, c=samples_all.iloc[:,301])
plt.xlabel("T1", fontsize=15)
plt.ylabel("T2", fontsize=15)
cbar = plt.colorbar()
cbar.set_label("PrismEXP score", fontsize=15)
plt.savefig("figures/tsne_features_heat.pdf")
plt.close()


plt.scatter(tt[:,0], tt[:,1], s=0.7, alpha=0.5, c=samples_all.iloc[:,300])
plt.xlabel("T1", fontsize=15)
plt.ylabel("T2", fontsize=15)
cbar = plt.colorbar()
cbar.set_label("global avg correlation", fontsize=15)
plt.savefig("figures/tsne_features_heat_global.pdf")
plt.close()

plt.figure(figsize=(24, 8))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.scatter(tt[:true_all.shape[0],0], tt[:true_all.shape[0],1], s=0.7, alpha=0.5, color="red")
ax1.scatter(tt[true_all.shape[0]:,0], tt[true_all.shape[0]:,1], s=0.7, alpha=0.5, color="black")
ax1.xlabel("T1", fontsize=15)
ax1.ylabel("T2", fontsize=15)
colors = {'true samples':'red', 'false samples':'black'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
ax1.legend(handles, labels)

ax2.scatter(tt[:,0], tt[:,1], s=0.7, alpha=0.5, c=samples_all.iloc[:,300])
ax2.xlabel("T1", fontsize=15)
ax2.ylabel("T2", fontsize=15)
cbar = ax2.colorbar()
cbar.set_label("global avg correlation", fontsize=15)

ax3.scatter(tt[:,0], tt[:,1], s=0.7, alpha=0.5, c=samples_all.iloc[:,301])
ax3.xlabel("T1", fontsize=15)
ax3.ylabel("T2", fontsize=15)
cbar = ax3.colorbar()
cbar.set_label("PrismEXP score", fontsize=15)

plt.savefig("figures/combined_featurespace.pdf")
plt.close()

