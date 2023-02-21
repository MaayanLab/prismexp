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

import matplotlib.pyplot as plt
import prismx as px

from sklearn.cluster import KMeans
import h5py as h5
import numpy as np
import pandas as pd
import random
from typing import List
import sys

from prismx.utils import quantile_normalize, normalize
from prismx.filter import hykGeneSelection


f = h5.File("mouse_matrix.h5", 'r')
expression = f['data/expression']
rsamples = sorted(random.sample(range(0, expression.shape[0]), 5000))
genes = f['meta/genes'][:]

subexp = expression[rsamples,:]
f.close()

f = h5.File("test_expression.h5", 'w')
f.create_dataset('data/expression', data=np.array(subexp))
f.create_dataset('meta/genes', data=np.array(genes))
f.create_dataset('meta/Sample_geo_accession', data=np.array(range(0, 5000)))
f.close()

ph5.show_h5("test_expression.h5")


f = h5.File("test_expression.h5"file, 'r')
expression = f['data/expression']
samples = f['meta/Sample_geo_accession']
genes = hykGeneSelection(h5file, geneidx)
genes.sort()
exp = 0     # keep memory footprint low
exp = expression[:, genes]
qq = normalize(exp, step_size=100, transpose=True)
kmeans = KMeans(n_clusters=clusterCount, random_state=42).fit(qq.transpose())
qq = 0      # keep memory footprint low
clustering = kmeans.labels_


h5file = "test_expression.h5"
outputFolder = "sink"

count = []
rr = np.power(np.array(range(1, 10)), 2)
for i in rr:
    print(i)
    count2 = []
    rr2 = np.array(range(1, 10))/250
    for j in rr2:
        print(j)
        filteredGenes = px.filter_genes(h5file, readThreshold=i, sampleThreshold=j, filterSamples=5000)
        count2.append(len(filteredGenes))
    count.append(count2)

df = pd.DataFrame(count)
df.columns = list(rr2)
df.index = list(rr)

filteredGenes = px.filterGenes(h5file, readThreshold=20, sampleThreshold=0.02, filterSamples=5000)

ax = sns.lineplot(data=df,markers=True, dashes=True)
ax.set_xlabel('read threshold', fontsize=15)
ax.set_ylabel('passing genes', fontsize=15)
plt.plot(20, len(filteredGenes), marker='o', color='orange', markersize=15, label="PrismEXP")
patch = mpatches.Patch(color='orange', label='PrismEXP')
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, loc='upper right')
ax.legend().set_title("sample threshold")
plt.setp(ax.get_legend().get_title(), fontsize='14')
plt.savefig("figures/threshold.pdf")
plt.close()



sns.lineplot(data=df)
plt.show()


count = []
rr = np.power(np.array(range(1, 5)), 2)
for i in rr:
    print(i)
    count2 = []
    rr2 = np.power(np.array(range(1, 5)), 2)/500
    for j in rr2:
        print(j)
        filteredGenes = px.filter_genes(h5file, readThreshold=i, sampleThreshold=j, filterSamples=5000)
        count2.append(len(filteredGenes))
    count.append(count2)

plt.tight_layout()
plt.plot(rr, count2, linestyle='--', marker='o', color='b')
plt.plot(0.01, count2[4], marker='o', color='orange', markersize=15)
plt.ylabel('passing genes', fontsize=16)
plt.xlabel('sample threshold', fontsize=16)
plt.savefig("figures/sample_threshold.pdf")
plt.close()








os.makedirs(outputFolder, exist_ok=True)
clustering = px.createClustering(h5file, filteredGenes, clusterGeneCount, clusterCount)

