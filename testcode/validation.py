import os
import time
import shutil
import sys
sys.path.append('C:/prismx/')

import h5py as h5
import pandas as pd
import numpy as np
import random
import show_h5 as ph5
import seaborn as sns
import matplotlib.patches as mpatches
from scipy import stats

import matplotlib.pyplot as plt

import prismx as px


f100 = pd.read_csv("logs/validationscore100.tsv", sep="\t")

pgene = stats.ttest_ind(f100["global_gene"], f100["prismx_gene"])[1]

sns.set(font_scale = 2)
f, ax = plt.subplots(figsize=(6, 6), frameon=True)
ax.grid(True)
ax.set_facecolor("white")
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
plt.tight_layout()
ax.scatter(f100["global_gene"], f100["prismx_gene"])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='r', ls='--')
ax.set_xlabel("global average AUC", fontsize=20)
ax.set_ylabel("PrismEXP average AUC", fontsize=20)
plt.savefig("figures/validation_gene.pdf")
plt.close()

plt.rcParams["axes.labelsize"] = 14
sns.set(font_scale = 3)
plt.tight_layout()
dd = pd.DataFrame({"global":f100["global_gene"], "PrismEXP":f100["prismx_gene"]})
ax = sns.violinplot(data=dd)
ax.set_ylabel("average AUC", fontsize=35)
plt.ylim(0.3, 1.2)
plt.plot([0, 1],[1.07, 1.07], 'k-', lw=2)
plt.text(-0.07, 1.1, "p-value: "+"{:.2e}".format(float(pgene)), fontsize=28)
plt.savefig("figures/validation_gene_violin.pdf",bbox_inches='tight')
plt.close()

pset = stats.ttest_ind(f100["global_set"], f100["prismx_set"])[1]
f, ax = plt.subplots(figsize=(6, 6))
ax.grid(True)
ax.set_facecolor("white")
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
plt.tight_layout()
ax.scatter(f100["global_set"], f100["prismx_set"])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='r', ls='--')
ax.set_xlabel("global average AUC", fontsize=20)
ax.set_ylabel("PrismEXP average AUC", fontsize=20)
plt.savefig("figures/validation_set.pdf")
plt.close()

plt.rcParams["axes.labelsize"] = 14
sns.set(font_scale = 3)
plt.tight_layout()
dd = pd.DataFrame({"global":f100["global_set"], "PrismEXP":f100["prismx_set"]})
ax = sns.violinplot(data=dd)
ax.set_ylabel("average AUC", fontsize=35)
plt.ylim(0.3, 1.2)
plt.plot([0, 1],[1.07, 1.07], 'k-', lw=2)
plt.text(-0.07, 1.1, "p-value: "+"{:.2e}".format(float(pset)), fontsize=28)
plt.savefig("figures/validation_set_violin.pdf",bbox_inches='tight')
plt.close()




f50 = pd.read_csv("logs/validationscore50.tsv", sep="\t")

pgene = stats.ttest_ind(f50["global_gene"], f50["prismx_gene"])[1]

sns.set(font_scale = 2)
f, ax = plt.subplots(figsize=(6, 6), frameon=True)
ax.grid(True)
ax.set_facecolor("white")
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
plt.tight_layout()
ax.scatter(f50["global_gene"], f50["prismx_gene"])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='r', ls='--')
ax.set_xlabel("global average AUC", fontsize=20)
ax.set_ylabel("PrismEXP average AUC", fontsize=20)
plt.savefig("figures/validation_gene_50.pdf")
plt.close()

plt.rcParams["axes.labelsize"] = 14
sns.set(font_scale = 3)
plt.tight_layout()
dd = pd.DataFrame({"global":f50["global_gene"], "PrismEXP":f50["prismx_gene"]})
ax = sns.violinplot(data=dd)
ax.set_ylabel("average AUC", fontsize=35)
plt.ylim(0.3, 1.2)
plt.plot([0, 1],[1.07, 1.07], 'k-', lw=2)
plt.text(-0.07, 1.1, "p-value: "+"{:.2e}".format(float(pgene)), fontsize=28)
plt.savefig("figures/validation_gene_violin_50.pdf",bbox_inches='tight')
plt.close()

pset = stats.ttest_ind(f50["global_set"], f50["prismx_set"])[1]
f, ax = plt.subplots(figsize=(6, 6))
ax.grid(True)
ax.set_facecolor("white")
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
plt.tight_layout()
ax.scatter(f50["global_set"], f50["prismx_set"])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='r', ls='--')
ax.set_xlabel("global average AUC", fontsize=20)
ax.set_ylabel("PrismEXP average AUC", fontsize=20)
plt.savefig("figures/validation_set_50.pdf")
plt.close()

plt.rcParams["axes.labelsize"] = 14
sns.set(font_scale = 3)
plt.tight_layout()
dd = pd.DataFrame({"global":f50["global_set"], "PrismEXP":f50["prismx_set"]})
ax = sns.violinplot(data=dd)
ax.set_ylabel("average AUC", fontsize=35)
plt.ylim(0.3, 1.2)
plt.plot([0, 1],[1.07, 1.07], 'k-', lw=2)
plt.text(-0.07, 1.1, "p-value: "+"{:.2e}".format(float(pset)), fontsize=28)
plt.savefig("figures/validation_set_violin_50.pdf",bbox_inches='tight')
plt.close()



diff50 = list(f50.iloc[:,2] - f50.iloc[:,1])
f50["dgene"] = diff50
diff50set = list(f50.iloc[:,4] - f50.iloc[:,3])
f50["dset"] = diff50set
np.corrcoef(diff50, diff50set)
o50 = np.argsort(diff50)[::-1]
f50.iloc[o50[61:100], :]


diff100 = list(f100.iloc[:,2] - f100.iloc[:,1])
f100["dgene"] = diff100
diff100set = list(f100.iloc[:,4] - f100.iloc[:,3])
f100["dset"] = diff100set
np.corrcoef(diff100, diff100set)
o100 = np.argsort(diff100)[::-1]
f100.iloc[o100[0:50], :]

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

f50.index = f50.iloc[:,0]
f100.index = f100.iloc[:,0]

inter = intersection(list(f50.iloc[:,0]), list(f100.iloc[:,0]))
inter.remove("GeneSigDB")
inter.remove("GeneSigDB")

import collections
a = f50.loc[inter, "GMT"]
print([item for item, count in collections.Counter(a).items() if count > 1])

np.corrcoef(f100.loc[inter,"dgene"], f50.loc[inter, "dgene"])

np.corrcoef(f100.loc[inter,"dgene"], f50.loc[inter, "dgene"])


list(f50.loc[inter, "GMT"])

geneimp50 = np.mean(f50.iloc[:,2]) - np.mean(f50.iloc[:,1])
setimp50 = np.mean(f50.iloc[:,4]) - np.mean(f50.iloc[:,3])

geneimp100 = np.mean(f100.iloc[:,2]) - np.mean(f100.iloc[:,1])
setimp100 = np.mean(f100.iloc[:,4]) - np.mean(f100.iloc[:,3])

print(str(geneimp50)+" - "+str(geneimp100))
print(str(setimp50)+" - "+str(setimp100))










pset = stats.ttest_ind(f100["global_set"], f100["prismx_set"])[1]
f, ax = plt.subplots(figsize=(6, 6))
ax.scatter(f100["global_set"], f100["prismx_set"])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='r', ls='--')
plt.savefig("figures/validation_set.pdf")

