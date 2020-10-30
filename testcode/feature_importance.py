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
import pandas as pd
from matplotlib.pyplot import *

model = pickle.load(open('gobp_model_q.pkl', 'rb'))

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1]



# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


plt.figure(figsize=(16,6))
gca().set_xticklabels(['']*10)
plt.plot([0, 0], [0, importances[300]*100], color="blue", linewidth=3)
plt.ylabel("Feature Importance (%)", fontsize=16)
plt.xlabel("Features (correlation matrix)", fontsize=16)
plt.bar(range(len(indices)), importances[indices]*100,
        color="r", yerr=std[indices]*100, align="center")
#plt.xticks(range(len(indices)), indices)
plt.xlim([-1, len(indices)])

colors = {'feature importance':'red', 'importance std':'black', 'global correlation':'blue'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)

plt.savefig("figures/feature_importance.pdf")


# calculate AUC scores for individual correlation matrices. 
aucs = pd.read_feather("testdata/full_auc_set.f")
aucs = aucs.set_index("index")
ma = aucs.mean()

plt.close()

plt.figure(2)
plt.scatter(importances*100, ma.iloc[0:301], c="black", s=6)
plt.scatter(importances[300]*100, ma.iloc[300], c="red", s=14)
plt.xlim(0,  1.6)
plt.ylim(0.45, 0.8)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("feature importance (%)", fontsize=18)
plt.ylabel("set AUC", fontsize=18)
plt.savefig("figures/auc_importance2.pdf")
plt.close()


# calculate AUC scores for individual correlation matrices. 
aucs = pd.read_feather("testdata/full_auc_gene.f")
aucs = aucs.set_index("index")
ma = aucs.mean()

plt.close()
plt.figure(2)
plt.scatter(importances*100, ma.iloc[0:301], c="black", s=6)
plt.scatter(importances[300]*100, ma.iloc[300], c="red", s=14)
plt.xlim(0,  1.6)
plt.ylim(0.45, 0.8)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("feature importance (%)", fontsize=18)
plt.ylabel("gene AUC", fontsize=18)
plt.savefig("figures/auc_importance_2.pdf")
plt.close()
