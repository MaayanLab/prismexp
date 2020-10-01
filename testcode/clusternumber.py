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

f5.index = f5.iloc[:,0]
f10.index = f10.iloc[:,0]
f25.index = f25.iloc[:,0]
f50.index = f50.iloc[:,0]
f100.index = f100.iloc[:,0]

inter = list(set(intersection(list(f50.iloc[:,0]), list(f100.iloc[:,0]))))
inter = list(set(intersection(inter, list(f25.iloc[:,0]))))
inter = list(set(intersection(inter, list(f10.iloc[:,0]))))
inter = list(set(intersection(inter, list(f5.iloc[:,0]))))

score5 = pd.DataFrame()
score10 = pd.DataFrame()
score25 = pd.DataFrame()
score50 = pd.DataFrame()
score100 = pd.DataFrame()
for i in range(0, len(inter)):
    score5[inter[i]] = f5.iloc[list(f5.index).index(inter[i]), :][1:]
    score10[inter[i]] = f10.iloc[list(f10.index).index(inter[i]), :][1:]
    score25[inter[i]] = f25.iloc[list(f25.index).index(inter[i]), :][1:]
    score50[inter[i]] = f50.iloc[list(f50.index).index(inter[i]), :][1:]
    score100[inter[i]] = f100.iloc[list(f100.index).index(inter[i]), :][1:]

pset = pd.DataFrame()
pset["global"] = score5.loc["global_set",:]
pset["5"] = score5.loc["prismx_set",:]
pset["10"] = score10.loc["prismx_set",:]
pset["25"] = score25.loc["prismx_set",:]
pset["50"] = score50.loc["prismx_set",:]
pset["100"] = score100.loc["prismx_set",:]


pgene = pd.DataFrame()
pgene["global"] = score5.loc["global_gene",:]
pgene["5"] = score5.loc["prismx_gene",:]
pgene["10"] = score10.loc["prismx_gene",:]
pgene["25"] = score25.loc["prismx_gene",:]
pgene["50"] = score50.loc["prismx_gene",:]
pgene["100"] = score100.loc["prismx_gene",:]


def func(x, a, b, c):
    return a * np.log(b * x) + c



avg_set = np.median(pset, axis=0)
popt, pcov = curve_fit(func, [1,5,10,25,50,100], avg_set)
xdata = np.linspace(1, 100, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt), linewidth=3)
xdata = np.linspace(1, 300, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt), linewidth=3, linestyle='dashed')
xd = np.array([1,5,10,25,50,100])
apy = func(xd, *popt)
plt.plot(xd, apy, 'o', color="black")
plt.ylabel("average set AUC", fontsize=20)
plt.xlabel("clusters", fontsize=20)
plt.savefig("figures/cluster_count_set_pred.pdf",bbox_inches='tight')
plt.close()


sns.set(font_scale = 3)
sns.set_theme(style="whitegrid")
plt.tight_layout()
ax = sns.violinplot(data=pset, cut=0.5)
ax.set_ylabel("average set AUC", fontsize=18)
ax.set_xlabel("clusters", fontsize=18)
plt.savefig("figures/cluster_count_set.pdf",bbox_inches='tight')
plt.close()


avg_gene = np.median(pgene, axis=0)
popt, pcov = curve_fit(func, [1,5,10,25,50,100], avg_gene)
xdata = np.linspace(1, 100, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt), linewidth=3)
xdata = np.linspace(1, 300, 1000)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt), linewidth=3, linestyle='dashed')
xd = np.array([1,5,10,25,50,100])
apy = func(xd, *popt)
plt.plot(xd, apy, 'o', color="black")
plt.ylabel("average gene AUC", fontsize=18)
plt.xlabel("clusters", fontsize=18)
plt.savefig("figures/cluster_count_gene_pred.pdf",bbox_inches='tight')
plt.close()



sns.set(font_scale = 3)
sns.set_theme(style="whitegrid")
plt.tight_layout()
ax = sns.violinplot(data=pgene, cut=0.5)
ax.set_ylabel("average gene AUC", fontsize=18)
ax.set_xlabel("clusters", fontsize=18)

plt.savefig("figures/cluster_count_gene.pdf",bbox_inches='tight')
plt.close()




gset = stats.ttest_ind(score50.loc["global_set",:], score100.loc["global_set",:])[1]
pset = stats.ttest_ind(score50.loc["prismx_set",:], score100.loc["prismx_set",:])[1]

ggene = stats.ttest_ind(score50.loc["global_gene",:], score100.loc["global_gene",:])[1]
pgene = stats.ttest_ind(score50.loc["prismx_gene",:], score100.loc["prismx_gene",:])[1]

plt.scatter(score50.loc["global_gene",:], score100.loc["global_gene",:])
plt.plot([0.5, 1], [0.5, 1], color='r', ls='--', lw=2)
plt.show()

