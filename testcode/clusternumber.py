import os
import time
import shutil
import sys

import h5py as h5
import pandas as pd
import numpy as np
import random

import seaborn as sns
import matplotlib.patches as mpatches
from scipy import stats

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

f5 = pd.read_csv("logs/validationscore5.txt", sep="\t")
f10 = pd.read_csv("logs/validationscore10.txt", sep="\t")
f25 = pd.read_csv("logs/validationscore25.txt", sep="\t")
f50 = pd.read_csv("logs/validationscore50.txt", sep="\t")
f100 = pd.read_csv("logs/validationscore100.txt", sep="\t")
f300 = pd.read_csv("logs/validationscore300.txt", sep="\t")

f5.index = f5.iloc[:,0]
f10.index = f10.iloc[:,0]
f25.index = f25.iloc[:,0]
f50.index = f50.iloc[:,0]
f100.index = f100.iloc[:,0]
f300.index = f300.iloc[:,0]

inter = list(set(intersection(list(f50.iloc[:,0]), list(f100.iloc[:,0]))))
inter = list(set(intersection(inter, list(f25.iloc[:,0]))))
inter = list(set(intersection(inter, list(f10.iloc[:,0]))))
inter = list(set(intersection(inter, list(f5.iloc[:,0]))))
inter = list(set(intersection(inter, list(f300.iloc[:,0]))))

score5 = pd.DataFrame()
score10 = pd.DataFrame()
score25 = pd.DataFrame()
score50 = pd.DataFrame()
score100 = pd.DataFrame()
score300 = pd.DataFrame()
for i in range(0, len(inter)):
    score5[inter[i]] = f5.iloc[list(f5.index).index(inter[i]), :][1:]
    score10[inter[i]] = f10.iloc[list(f10.index).index(inter[i]), :][1:]
    score25[inter[i]] = f25.iloc[list(f25.index).index(inter[i]), :][1:]
    score50[inter[i]] = f50.iloc[list(f50.index).index(inter[i]), :][1:]
    score100[inter[i]] = f100.iloc[list(f100.index).index(inter[i]), :][1:]
    score300[inter[i]] = f300.iloc[list(f300.index).index(inter[i]), :][1:]

pset = pd.DataFrame()
pset["global"] = score5.loc["global_set",:]
pset["5"] = score5.loc["prismx_set",:]
pset["10"] = score10.loc["prismx_set",:]
pset["25"] = score25.loc["prismx_set",:]
pset["50"] = score50.loc["prismx_set",:]
pset["100"] = score100.loc["prismx_set",:]
pset["300"] = score300.loc["prismx_set",:]

pgene = pd.DataFrame()
pgene["global"] = score5.loc["global_gene",:]
pgene["5"] = score5.loc["prismx_gene",:]
pgene["10"] = score10.loc["prismx_gene",:]
pgene["25"] = score25.loc["prismx_gene",:]
pgene["50"] = score50.loc["prismx_gene",:]
pgene["100"] = score100.loc["prismx_gene",:]
pgene["300"] = score300.loc["prismx_gene",:]


pgene.to_csv("test_data/gene_auc5.tsv", sep="\t")
pset.to_csv("test_data/set_auc5.tsv", sep="\t")

def func(x, a, b, c):
    return a * np.log(b * x) + c

def power_law(x, a, b, c):
    return a * np.exp(b * x) + c

avg_set = list(np.mean(pset, axis=0))

med_set = list(np.median(pset, axis=0))
med_gene = list(np.median(pgene, axis=0))


#boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3'])



def func(x, a, b, c):
    return a * np.log(b * x) + c

med_gene_t = med_gene
med_gene_t.append(med_gene[-1])
med_gene_t.append(med_gene[-1])


med_set_t = med_set
med_set_t.append(med_set[-1])
med_set_t.append(med_set[-1])

med_set = list(np.median(pset, axis=0))
med_gene = list(np.median(pgene, axis=0))




popt, pcov = curve_fit(func, [1,5,10,25,50,100,300,300,300], med_set)
xdata = np.linspace(1, 300, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt), linewidth=3)
xdata = np.linspace(1, 500, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt), linewidth=3, linestyle='dashed')
xd = np.array([1,5,10,25,50,100,300])
apy = func(xd, *popt)
#plt.plot(xd, apy, 'o', color="black")
plt.plot(xd, med_set, 'o', color="black")
plt.ylabel("set AUC", fontsize=20)
plt.xlabel("clusters", fontsize=20)
plt.show()







popt, pcov = curve_fit(func, [1,5,10,25,50,100,300], med_gene)
xdata = np.linspace(1, 300, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt), linewidth=3)
xdata = np.linspace(1, 500, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt), linewidth=3, linestyle='dashed')
xd = np.array([1,5,10,25,50,100,300])
apy = func(xd, *popt)
#plt.plot(xd, apy, 'o', color="black")
plt.plot(xd, med_gene, 'o', color="black")
plt.ylabel("set AUC", fontsize=20)
plt.xlabel("clusters", fontsize=20)
plt.savefig("figures/median_cluster_count_set_pred3.pdf",bbox_inches='tight')
plt.close()


sns.set(font_scale = 3)
sns.set_theme(style="whitegrid")
plt.tight_layout()
ax = sns.violinplot(data=pset, cut=0.5)
ax.set_ylabel("set AUC", fontsize=18)
ax.set_xlabel("clusters", fontsize=18)
plt.savefig("figures/cluster_count_set3.pdf",bbox_inches='tight')
plt.close()


avg_gene = np.median(pgene, axis=0)
popt, pcov = curve_fit(func, [1,5,10,25,50,100,300], avg_gene)
xdata = np.linspace(1, 300, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt), linewidth=3)
xdata = np.linspace(1, 500, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt), linewidth=3, linestyle='dashed')
xd = np.array([1,5,10,25,50,100,300])
apy = func(xd, *popt)


plt.plot(xd, avg_gene, 'o', color="black")
plt.ylabel("average gene AUC", fontsize=18)
plt.xlabel("clusters", fontsize=18)
plt.savefig("figures/cluster_count_gene_pred3.pdf",bbox_inches='tight')
plt.close()





avg_set = list(np.mean(pset, axis=0))
popt, pcov = curve_fit(func, [1,5,10,25,50,100,300], med_set)
xdata = np.linspace(1, 300, 1000)
plt.plot(xdata, func(xdata, *popt), 'g-', linewidth=3)
xdata = np.linspace(1, 500, 1000)
plt.plot(xdata, func(xdata, *popt), 'g-', linewidth=3, linestyle='dashed')
xd = np.array([1,5,10,25,50,100,300])
apy = func(xd, *popt)
plt.plot(xd, med_set, 'o', color="green", label="set AUC")


avg_gene = np.median(pgene, axis=0)
popt, pcov = curve_fit(func, [1,5,10,25,50,100,300], med_gene)
xdata = np.linspace(1, 300, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-', linewidth=3)
xdata = np.linspace(1, 500, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-', linewidth=3, linestyle='dashed')
xd = np.array([1,5,10,25,50,100,300])
apy = func(xd, *popt)
plt.plot(xd, med_gene, 'o', color="red", label="gene AUC")

plt.ylabel("AUC", fontsize=18)
plt.xlabel("clusters", fontsize=18)
plt.legend()
plt.savefig("figures/cluster_count_combo_pred3.pdf",bbox_inches='tight')
plt.close()






sns.set(font_scale = 3)
sns.set_theme(style="whitegrid")
plt.tight_layout()
ax = sns.violinplot(data=pgene, cut=0.5)
ax.set_ylabel("gene AUC", fontsize=18)
ax.set_xlabel("clusters", fontsize=18)

plt.savefig("figures/cluster_count_gene3.pdf",bbox_inches='tight')
plt.close()

plt.close()
plt.scatter(pset.iloc[:,4], pset.iloc[:,6], s=1)
plt.plot([0.5,1], [0.5,1], color='r', ls='--', lw=2)
plt.savefig("figures/testpset.pdf")

dd = pset.iloc[:,5] - pset.iloc[:,6]
ds = dd.sort_values()
ds[0:10]
ds = dd.sort_values(ascending=False)
ds[0:10]

gset = stats.ttest_ind(score50.loc["global_set",:], score100.loc["global_set",:])[1]
pset = stats.ttest_ind(score50.loc["prismx_set",:], score100.loc["prismx_set",:])[1]

ggene = stats.ttest_ind(score50.loc["global_gene",:], score100.loc["global_gene",:])[1]
pgene = stats.ttest_ind(score50.loc["prismx_gene",:], score100.loc["prismx_gene",:])[1]

plt.scatter(score50.loc["global_gene",:], score100.loc["global_gene",:])
plt.plot([0.5, 1], [0.5, 1], color='r', ls='--', lw=2)
plt.show()

pval_gene = stats.ttest_rel(pgene.loc[:,"global"],pgene.loc[:,"300"])[1]
pval_set = stats.ttest_rel(pset.loc[:,"global"],pset.loc[:,"300"])[1]
