import pandas as pd
import numpy as np
import os
import math
import random
import pickle
import time
import feather

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

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture
from sklearn.metrics.cluster import homogeneity_score
from scipy import stats

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

k = 60
clusterer = KMeans(n_clusters=k, random_state=10)
cluster_labels = clusterer.fit_predict(samples_all.iloc[:,0:301])

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

plt.scatter(tt[:,0], tt[:,1], s=0.7, alpha=0.5, c=colors[cluster_labels])
plt.savefig("figures/cluster_features_high.pdf")
plt.close()


truth = [1]*true_all.shape[0]
res = truth+([0]*(tt.shape[0]-len(truth)))

homogeneity_score(cluster_labels, samples_all.iloc[:,301])

homogeneity_score(cluster_labels, samples_all.iloc[:,300])


stats.ttest_ind(samples_all.iloc[:true_all.shape[0],301], samples_all.iloc[true_all.shape[0]:,301], equal_var = False)
stats.ttest_ind(samples_all.iloc[:true_all.shape[0],300], samples_all.iloc[true_all.shape[0]:,300], equal_var = False)

pk = pd.DataFrame()
pk["pred"] = samples_all.iloc[:,301]
pk["lab"] = res

for cl in [1,0]:
    # Subset to the airline
    subset = pk[pk['lab'] == cl]
    # Draw the density plot
    sns.distplot(subset['pred'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3},
                 label = cl)

# Plot formatting
plt.legend(prop={'size': 16}, title = 'Class')
plt.xlabel('PrismEXP score')
plt.ylabel('Density')
plt.savefig("figures/density_pred_prismx.pdf")
plt.close()



pk = pd.DataFrame()
pk["pred"] = samples_all.iloc[:,0]
pk["lab"] = res

for cl in [1,0]:
    # Subset to the airline
    subset = pk[pk['lab'] == cl]
    # Draw the density plot
    sns.distplot(subset['pred'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3},
                 label = cl)

# Plot formatting
plt.legend(prop={'size': 16}, title = 'Class')
plt.xlabel('global score')
plt.ylabel('Density')
plt.savefig("figures/density_pred_global.pdf")
plt.close()


[g]


colp = sns.color_palette("hls", 60)

plt.scatter(tt[:,0], tt[:,1], s=0.7, alpha=0.5, c=colp[cluster_labels])
plt.savefig("figures/cluster_features_high.pdf")
plt.close()


for k in range(30, 60):
    clusterer = KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusterer.fit_predict(tt)
    silhouette_avg = silhouette_score(tt, cluster_labels)
    print(silhouette_avg)


plt.figure(figsize=(7,6))
plt.scatter(tt[:true_all.shape[0],0], tt[:true_all.shape[0],1], s=0.7, alpha=0.5, color="red")
plt.scatter(tt[true_all.shape[0]:,0], tt[true_all.shape[0]:,1], s=0.7, alpha=0.5, color="black")
plt.xlabel("T1", fontsize=15)
plt.ylabel("T2", fontsize=15)
colors = {'true samples':'red', 'false samples':'black'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.savefig("figures/tsne_features2.pdf")
plt.close()

sc = samples_all.iloc[:,301]
sc = sc.fillna(0)

plt.scatter(tt[:,0], tt[:,1], s=0.7, alpha=0.5, c=samples_all.iloc[:,301])
plt.xlabel("T1", fontsize=15)
plt.ylabel("T2", fontsize=15)
cbar = plt.colorbar()
cbar.set_label("PrismEXP score", fontsize=15)
plt.savefig("figures/tsne_features_heat4.pdf")
plt.close()


plt.scatter(tt[:,0], tt[:,1], s=0.7, alpha=0.5, c=samples_all.iloc[:,300])
plt.xlabel("T1", fontsize=15)
plt.ylabel("T2", fontsize=15)
cbar = plt.colorbar()
cbar.set_label("global avg correlation", fontsize=15)
plt.savefig("figures/tsne_features_heat_global4.pdf")
plt.close()


plt.figure(figsize=(7, 6))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.scatter(tt[:true_all.shape[0],0], tt[:true_all.shape[0],1], s=0.7, alpha=0.5, color="red")
ax1.scatter(tt[true_all.shape[0]:,0], tt[true_all.shape[0]:,1], s=0.7, alpha=0.5, color="black")
plt.xlabel("T1", fontsize=15)
plt.ylabel("T2", fontsize=15)
colors = {'true samples':'red', 'false samples':'black'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels, ax=ax1)

ok1 = ax2.scatter(tt[:,0], tt[:,1], s=0.7, alpha=0.5, c=samples_all.iloc[:,300])
plt.xlabel("T1", fontsize=15)
plt.ylabel("T2", fontsize=15)
cbar = plt.colorbar(ok1, ax=ax2)
cbar.set_label("global avg correlation", fontsize=15)

ok2 = ax3.scatter(tt[:,0], tt[:,1], s=0.7, alpha=0.5, c=samples_all.iloc[:,301])
plt.xlabel("T1", fontsize=15)
plt.ylabel("T2", fontsize=15)
cbar = plt.colorbar(ok2, ax=ax3)
cbar.set_label("PrismEXP score", fontsize=15)

plt.savefig("figures/combined_featurespace2.pdf")
plt.close()


clusterer = mixture.GaussianMixture(n_components=n_clusters)
#clusterer = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = clusterer.fit_predict(X)

silhouette_avg = silhouette_score(X, cluster_labels)



pp = pd.DataFrame()
K = [10,12,14,16,18,20,22,24,26]
for k in K:
    sc = []
    for i in range(10):
        clusterer = mixture.GaussianMixture(n_components=n_clusters)
        #clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        sc.append(silhouette_score(X, cluster_labels))
    print(sc)
    pp[k] = sc






X = tt
n_clusters = 26

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
ax1.set_xlim([-0.1, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
# Initialize the clusterer with n_clusters value and a random generator
# seed of 10 for reproducibility.
clusterer = KMeans(n_clusters=n_clusters, random_state=10)
clusterer = mixture.GaussianMixture(n_components=n_clusters, random_state=10)
cluster_labels = clusterer.fit_predict(X)
# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(X, cluster_labels)
print("For n_clusters =", n_clusters,
        "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, cluster_labels)
y_lower = 10
for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)
    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")
# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
# 2nd Plot showing the actual clusters formed
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
            c=colors, edgecolor='k')

# Labeling the clusters
centers = clusterer.cluster_centers_
# Draw white circles at cluster centers
ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=200, edgecolor='k')
for i, c in enumerate(centers):
    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                s=50, edgecolor='k')

ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")
plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                "with n_clusters = %d" % n_clusters),
                fontsize=14, fontweight='bold')

plt.savefig("figures/silhouette.pdf")

