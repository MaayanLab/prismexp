import urllib.request
import prismx as px
import pickle
from memory_profiler import memory_usage
import os
import time
import matplotlib

t1 = 0
t2 = 0
t3 = 0

def runPrismX(clusterCount: int):
    #urllib.request.urlretrieve("https://mssm-seq-matrix.s3.amazonaws.com/mouse_matrix.h5", "mouse_matrix.h5")
    start = time.time()
    correlationFolder = "correlation_"+str(clusterCount)+"_folder_q"
    predictionFolder = "prediction_"+str(clusterCount)+"_folder_q"
    libs = px.list_libraries()
    gmt_file = px.load_library(libs[110])
    px.createCorrelationMatrices("mouse_matrix.h5", correlationFolder, clusterCount=clusterCount, sampleCount=5000, correlationSampleCount=5000, verbose=True)
    t1 = time.time()-start
    print("T1: "+str(t1))
    px.correlation_scores(gmt_file, correlationFolder, predictionFolder, verbose=True)
    t2 = time.time()-start
    print("T2: "+str(t2))
    model = px.trainModel(predictionFolder, correlationFolder, gmt_file, training_size=300000, test_train_split=0.1, sample_positive=40000, sample_negative=200000, random_state=42, verbose=True)
    pickle.dump(model, open("gobp_model_"+str(clusterCount)+".pkl", 'wb'))
    t3 = time.time() - start
    print("T3: "+str(t3))

mem_usage = memory_usage(runPrismX, interval=1)
pickle.dump(mem_usage, open("mem_usage.pkl", "wb"))
print('Maximum memory usage: %s' % str(max(mem_usage)))


import numpy as np
import matplotlib.pyplot as plt
import pickle

mem_usage = np.asarray(pickle.load(open("mem_usage.pkl", "rb")))/1000

filterTime = round(5.5*60)
clusterSamples = round(34.1*60) + filterTime
calculateCorrelation = 18167
predictionTime = 25307

xscale = 3600

f = plt.figure(figsize=[14,6])
plt.plot(np.asarray(range(0, filterTime))/xscale, mem_usage[0:filterTime], label="gene filter")
plt.plot(np.asarray(range(filterTime, clusterSamples))/xscale, mem_usage[filterTime:clusterSamples], label="sample clustering")
plt.plot(np.asarray(range(clusterSamples, calculateCorrelation))/xscale, mem_usage[clusterSamples:calculateCorrelation], label="correlation")
plt.plot(np.asarray(range(calculateCorrelation, predictionTime))/xscale, mem_usage[calculateCorrelation:predictionTime], label="prediction")
plt.plot(np.asarray(range(predictionTime, len(mem_usage)))/xscale, mem_usage[predictionTime:len(mem_usage)], label="training")

plt.xlabel('runtime (hours)', fontsize=14)
plt.ylabel('memory consumption (GB)', fontsize=14)
#plt.show()
plt.legend(framealpha=0, frameon=False, fontsize=14)

f.savefig("memroy_profiler_100.pdf", bbox_inches='tight')


filteredGenes = filter_genes("mouse_matrix.h5", 20, 0.01, 5000)
genes = hykGeneSelection("mouse_matrix.h5", filteredGenes)



import urllib.request
import prismx as px
import pickle
from memory_profiler import memory_usage
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import feather
from prismx.utils import load_json, get_config, get_data_path, read_gmt
from prismx.loaddata import get_genes
import numpy as np

px.print_libraries()
libs = px.list_libraries()
gmt_file = px.load_library(libs[28])

outname = libs[28]
correlationFolder = "correlation_100_folder"
predictionFolder = "prediction_100_folder"
outfolder = "prismxresult_100"


px.predict_gmt("gobp_model_100.pkl", gmt_file, correlationFolder, predictionFolder, outfolder, outname, step_size=1000, verbose=True)

geneAUC, setAUC = px.benchmarkGMT(gmt_file, correlationFolder, predictionFolder, outfolder+"/"+outname+".f", verbose=True)


fig, ax = plt.subplots(figsize=[4,16])
violin_parts = ax.violinplot(setAUC.values, range(0,setAUC.shape[1]), points=200, vert=False, widths=0.7, showextrema=False, showmedians=True)
ax.set_title('GWAS', fontsize=14)
ax.set_yticks(range(0,len(geneAUC.columns)))
ax.set_yticklabels(geneAUC.columns)
ax.set_xlabel("AUC")
plt.xlim([0.5,1])

ax.get_children()[51].set_color('r')
ax.get_children()[50].set_color('b')
for i in range(0, 50):
    ax.get_children()[i].set_color('#000000')

vp = violin_parts['cmedians']
vp.set_edgecolor('#000000')
vp.set_linewidth(2)

plt.axvline(np.median(setAUC["prismx"]), 0, 1, color="#ee2222", linewidth=1, linestyle=(0,(4,4)))
plt.savefig(outname+'_100.png')
#plt.show()

plt.close()



import prismx as px

correlationFolder = "correlation_folder_300_q"
predictionFolder = "prediction_folder_300_q"

# download/initialize gmt file
px.print_libraries()
libs = px.list_libraries()
gmt_file = px.load_library(libs[111])
clusterCount = 300

# apply gene filtering, expression clustering and correlation calculations
px.createCorrelationMatrices("mouse_matrix.h5", correlationFolder, clusterCount=clusterCount, sampleCount=5000, 
            correlationSampleCount=5000, verbose=True)

# average correlation scores for a given gmt file
px.correlation_scores(gmt_file, correlationFolder, predictionFolder, verbose=True)

# build a training data set and train model
model = px.trainModel(predictionFolder, correlationFolder, gmt_file, training_size=300000, 
            test_train_split=0.1, sample_positive=40000, sample_negative=200000, random_state=42, verbose=True)

pickle.dump(model, open("gobp_model_"+str(clusterCount)+"_q.pkl", 'wb'))


outfolder = "prismxresult_q"


gmt_file = px.load_library(libs[111])
outname = libs[111]

# calculate PrismX predictions with pretrained model
px.predict_gmt("gobp_model_q.pkl", gmt_file, correlationFolder, predictionFolder, outfolder, outname, step_size=1000, verbose=False)
# benchmark the prediction quality
geneAUC, setAUC = px.benchmarkGMTfast(gmt_file, correlationFolder, predictionFolder, outfolder+"/"+outname+".f", verbose=True)



gmt_file = px.load_library(libs[43])
outname = libs[43]


# calculate PrismX predictions with pretrained model
px.predict_gmt("gobp_model_q.pkl", gmt_file, correlationFolder, predictionFolder, outfolder, outname, step_size=1000, verbose=False)
# benchmark the prediction quality
geneAUC, setAUC = px.benchmarkGMTfast(gmt_file, correlationFolder, predictionFolder, outfolder+"/"+outname+".f", verbose=True)

