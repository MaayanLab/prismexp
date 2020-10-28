
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
from sklearn.metrics import roc_auc_score, roc_curve, auc

import prismx as px

from prismx.utils import readGMT, loadCorrelation, loadPrediction
from prismx.prediction import correlationScores, loadPredictionsRange
from sklearn.manifold import TSNE


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2]
    return lst3 

def rank(arr):       
    sorted_list = sorted(arr)
    rank = 1
    # seed initial rank as 1 because that's first item in input list
    sorted_rank_list = [1]
    for i in range(1, len(sorted_list)):
        if sorted_list[i] != sorted_list[i-1]:
            rank += 1
        sorted_rank_list.append(rank)
    rank_list = []
    for item in arr:
        for index, element in enumerate(sorted_list):
            if element == item:
                rank_list.append(sorted_rank_list[index])
                # we want to break out of inner for loop
                break
    return rank_list

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

autorif = pd.read_csv("testdata/autophagy_autorif.tsv", sep="\t")
generif = pd.read_csv("testdata/autophagy_generif.tsv", sep="\t")

ff = list(filter(lambda x:'AUTOPHAG' in x, setAUC.index))

setAUC.loc[ff,:]

up = loadPrediction("prediction_folder_300_umap", "global")

idx = [x.decode("UTF-8") for x in up.index]

dict, rdict, ugenes = px.readGMT(gmtFile, backgroundGenes=idx)
dgenes = ugenes
ugenes = [x.encode("UTF-8") for x in ugenes]


ll = list(range(1))
ll.append("global")



aucs = list()
features = pd.DataFrame()
setname = list()
for i in ll:
    print(i)
    pk = pd.read_feather("correlation_folder_300/correlation_"+str(i)+".f")
    pk = pk.set_index("index")
    pk = pk.loc[ugenes,dgenes]
    aucs = list()
    setname = list()
    for f in list(dict.keys()):
        mm = pk.loc[:, dict[f]].mean(axis=1).fillna(0)
        gold = [x in dict[f] for x in pk.columns]
        fpr, tpr, _ = roc_curve(list(gold), list(mm))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    features[i] = aucs

features.index = list(dict.keys())

gop.loc[:,ff]


pglobal = pd.read_feather("prediction_folder_300_umap/prediction_global.f").set_index("index").loc[:,ff]
pglobal.index = [x.decode("UTF-8") for x in pglobal.index]

generif = pd.read_csv("testdata/autophagy_generif.tsv", sep="\t")
generif = generif.set_index("Gene")
autorif = pd.read_csv("testdata/autophagy_autorif.tsv", sep="\t")
autorif = autorif.set_index("Gene")


crisp = pd.read_csv("testdata/crispr_screen.tsv", sep="\t").iloc[:,1]
crispr = crisp.iloc[:,2]
crispr.index = crisp.iloc[:,0]

cs1 = "journal.pbio.2007044.s010.tsv"
cs2 = "jcb_201804132_tables1.tsv"
cs3 = "embr201845889-sup-0002-datasetev1.tsv"
cs4 = "elife-50034-supp8-v2.tsv"
cs5 = "elife-17290-fig4-data1-v1.tsv"

cs_path = "testdata/crispr_screen/"

#=============== 1 ===============

cs1 = "journal.pbio.2007044.s010.tsv"
cc1 = pd.read_csv(cs_path+cs1, sep="\t")
g1 = list(cc1.iloc[:,0])
v1 = list(cc1.iloc[:,5])
cc1 = pd.DataFrame()
cc1["score"] = v1
cc1.index = g1
cc1_t = pd.DataFrame()
cc1_t["score"] = list(cc1.iloc[:,0])
cc1_t.index = g1
cc1 = cc1_t.astype(np.float)
cc1 = cc1.groupby(cc1.index).mean()
cc1.to_csv("cleancrisp/crispr1.tsv", sep="\t")

#=============== 2 ===============

cs2 = "jcb_201804132_tables1.tsv"
cc2 = pd.read_csv(cs_path+cs2, sep="\t")
g1 = list(cc2.iloc[:,0])
vt1 = np.log2(np.array(cc2.iloc[:,3])+1)
vt2 = np.log2(np.array(cc2.iloc[:,4])+1)
v2 = vt2-vt1
cc2 = pd.DataFrame()
cc2["score"] = v2
cc2["gene"] = g1
cc2.index = g1
ct = cc2.groupby(['gene']).mean()
gg1 = list(ct.iloc[:,0].index)
cc2_1 = pd.DataFrame()
cc2_1["score"] = list(ct.iloc[:,0])
cc2_1.index = gg1

cc2 = pd.read_csv(cs_path+cs2, sep="\t")
g1 = list(cc2.iloc[:,0])
vt1 = np.log2(np.array(cc2.iloc[:,5])+1)
vt2 = np.log2(np.array(cc2.iloc[:,6])+1)
v2 = vt2-vt1
cc2 = pd.DataFrame()
cc2["score"] = v2
cc2["gene"] = g1
cc2.index = g1
ct = cc2.groupby(['gene']).mean()
gg1 = list(ct.iloc[:,0].index)
cc2_2 = pd.DataFrame()
cc2_2["score"] = list(ct.iloc[:,0])
cc2_2.index = gg1

np.corrcoef(list(cc2_1.iloc[:,0]),list(cc2_2.iloc[:,0]))
cc2_a = pd.DataFrame()
cc2_a["r1"] = list(cc2_1.iloc[:,0])
cc2_a["r2"] = list(cc2_2.iloc[:,0])
cc2 = cc2_a.mean(axis=1)
cc2.index = gg1
cc2 = cc2.to_frame().astype(np.float)
cc2 = cc2.groupby(cc2.index).mean()
cc2.to_csv("cleancrisp/crispr2.tsv", sep="\t")

#============= 3 ==============

cs3 = "embr201845889-sup-0002-datasetev1.tsv"
cc3 = pd.read_csv(cs_path+cs3, sep="\t")
cc3.iloc[1:,[0,2,3,4,5]]
gg1 = cc3.iloc[:,0]
v1 = cc3.iloc[:,1:].mean(axis=1)
v1.index = gg1
cc3 = v1
cc3 = cc3.to_frame().astype(np.float)
cc3 = cc3.groupby(cc3.index).mean()
cc3.to_csv("cleancrisp/crispr3.tsv", sep="\t")

#============= 4 ==============

cs4 = "elife-50034-supp2-v2.txt"
#cs4 = "elife-50034-supp8-v2.tsv"
cc4 = pd.read_csv(cs_path+cs4, sep="\t")
cc4_t = cc4.iloc[:,7]
cc4_t.index = cc4.iloc[:,0]
tt = pd.DataFrame()
tt["value"] = list(cc4.iloc[:,7])
tt["genes"] = list(cc4_t.index)
cc4 = cc4_t
cc4 = cc4.to_frame().astype(np.float)
cc4 = cc4.groupby(cc4.index).mean()
cc4.to_csv("cleancrisp/crispr4.tsv", sep="\t")

#============= 5 ==============

cs5 = "elife-17290-fig4-data1-v1.tsv"
cc5 = pd.read_csv(cs_path+cs5, sep="\t")
cc5_t = cc5.iloc[:,2]
cc5_t.index = cc5.iloc[:,1]
cc5 = cc5_t
cc5 = cc5.to_frame().astype(np.float)
cc5 = cc5.groupby(cc5.index).mean()
cc5.to_csv("cleancrisp/crispr5.tsv", sep="\t")
# =============================

screen = [cc1, cc2, cc3, cc4, cc5]

zero_data = np.zeros(shape=(len(screen),len(screen)))
c_screens = pd.DataFrame(zero_data, columns=[1,2,3,4,5])

for i in range(5):
    for j in range(5):
        print(str(i)+" - "+str(j))
        inter = intersection(list(screen[i].index), list(screen[j].index))
        cc = np.corrcoef(np.squeeze(screen[i].loc[inter]), np.squeeze(screen[j].loc[inter]))[0][1]
        c_screens.iloc[i,j] = cc

corlist = list()

dict["allautophagy"]

for f in ff:
    truepos = pd.DataFrame()
    dat = gop.loc[:, f].sort_values(ascending=False)
    ii = list(dat.index)
    isin = [x in dict[f] for x in ii]
    isin = [int(x) for x in isin]
    rdat = rank(list(dat))
    rdat = np.array(rdat)/max(rdat)
    gg = pglobal.loc[ii, f]
    rgg = rank(list(gg))
    rgg = np.array(rgg)/max(rgg)
    imp = pd.DataFrame()
    imp["prismx"] = dat
    imp["prismx_index"] = rdat
    imp["global"] = gg
    imp["global_index"] = rgg
    imp["gold"] = isin
    imp["autorif_count"] = autorif.loc[imp.index,:].iloc[:,1].fillna(0).astype("i")
    imp["autorif_specificity"] = autorif.loc[imp.index,:].iloc[:,2].fillna(0)
    imp["generif_count"] = generif.loc[imp.index,:].iloc[:,1].fillna(0).astype("i")
    imp["generif_specificity"] = generif.loc[imp.index,:].iloc[:,2].fillna(0)
    print("-------------------")
    print(f)
    print(dict[f])
    inter = intersection(crispr.index, imp.index)
    inter.sort()
    c1 = crispr.loc[inter]
    a1 = imp.loc[inter,:].iloc[:,1]
    a2 = imp.loc[inter,:].iloc[:,3]
    nn = np.corrcoef(c1,a1)
    nn2 = np.corrcoef(c1,a2)
    print("crispr vs prismx: "+str(nn[0][1]))
    print("crispr vs global: "+str(nn2[0][1]))
    print("prismx auc: "+str(setAUC.loc[f,:][1]))
    ii = imp.index[0:100]
    inter = intersection(list(screen[2].index), list(ii))
    print(inter)
    ccl = list()
    zero_data = np.zeros(shape=(5,9))
    cscore = pd.DataFrame(zero_data, columns=["prismx_auc", "gold_auc", "prismx", "prismx_i", "glob", "glob_i", "gold", "autorif", "autorif_spec"])
    for i in range(5):
        inter = intersection(list(screen[i].index), list(imp.index))
        gogo = [x in ii for x in inter]
        gogo2 = [x in dict[f] for x in inter]
        fpr, tpr, _ = roc_curve(list(gogo), np.squeeze(screen[i].loc[inter]))
        roc_auc = round(auc(fpr, tpr),4)
        fpr, tpr, _ = roc_curve(list(gogo2), np.squeeze(screen[i].loc[inter]))
        roc_auc2 = round(auc(fpr, tpr),4)
        cor1 = round(np.corrcoef(np.squeeze(screen[i].loc[inter]), np.squeeze(imp.loc[inter].iloc[:,0]))[0][1], 4)
        cor2 = round(np.corrcoef(np.squeeze(screen[i].loc[inter]), np.squeeze(imp.loc[inter].iloc[:,1]))[0][1], 4)
        cor3 = round(np.corrcoef(np.squeeze(screen[i].loc[inter]), np.squeeze(imp.loc[inter].iloc[:,2]))[0][1], 4)
        cor4 = round(np.corrcoef(np.squeeze(screen[i].loc[inter]), np.squeeze(imp.loc[inter].iloc[:,3]))[0][1], 4)
        cor5 = round(np.corrcoef(np.squeeze(screen[i].loc[inter]), np.squeeze(imp.loc[inter].iloc[:,4]))[0][1], 4)
        cor6 = round(np.corrcoef(np.squeeze(screen[i].loc[inter]), np.squeeze(imp.loc[inter].iloc[:,5]))[0][1], 4)
        cor7 = round(np.corrcoef(np.squeeze(screen[i].loc[inter]), np.squeeze(imp.loc[inter].iloc[:,6]))[0][1], 4)
        cors = [roc_auc, roc_auc2, cor1, cor2, cor3, cor4, cor5, cor6, cor7]
        cscore.iloc[i,:] = cors
        corlist.append(cscore)
    print(cscore)


    #print(imp.iloc[1:30,:])
    #imp.to_csv("testdata/"+f.replace(" ","_").replace(":","_")+".tsv", sep="\t")




inter = intersection(crispr.index, imp.index)
inter.sort()

c1 = crispr.loc[inter]
a1 = imp.loc[inter,:].iloc[:,1]

nn = np.corrcoef(c1,a1)


