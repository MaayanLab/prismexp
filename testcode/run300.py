import urllib.request
import prismx as px
import pickle
import os
import time
import matplotlib
import sys
import shutil


correlationFolder = "correlation_folder_300"
predictionFolder = "prediction_folder_300"
outfolder = "prismxresult"

clustn = 300

#px.createCorrelationMatrices("mouse_matrix.h5", correlationFolder, clusterCount=clustn, sampleCount=5000, correlationSampleCount=5000, verbose=True)

libs = px.listLibraries()
gmtFile = px.loadLibrary(libs[111], overwrite=True)
#px.correlationScores(gmtFile, correlationFolder, predictionFolder, verbose=True)

#model = px.trainModel(predictionFolder, correlationFolder, gmtFile, trainingSize=300000, testTrainSplit=0.1, samplePositive=40000, sampleNegative=200000, randomState=42, verbose=True)
#pickle.dump(model, open("gobp_model_300.pkl", 'wb'))


f = open("validationscore"+str(clustn)+".txt", 'r')
libraries = [x.split("\t")[0] for x in f.readlines()]
newlibs = list(set(libs).difference(set(libraries)))
newlibs.sort()
#newlibs.reverse()
newlibs = newlibs[50:60]

for i in range(0, len(newlibs)):
    try:
        print(newlibs[i])
        gmtFile = px.loadLibrary(newlibs[i])
        print("loaded")
        g1, g2, g3 = px.readGMT(gmtFile)
        # set output configuration
        outname = newlibs[i]
        if len(g1) < 14000:
            # calculate PrismX predictions with pretrained model
            px.predictGMT("gobp_model_"+str(clustn)+".pkl", gmtFile, correlationFolder, predictionFolder, outfolder, outname, stepSize=200, intersect=True, verbose=True)
            # benchmark the prediction quality
            geneAUC, setAUC = px.benchmarkGMTfast(gmtFile, correlationFolder, predictionFolder, outfolder+"/"+outname+".f", intersect=True, verbose=True)
            gv = geneAUC.iloc[:,1].mean()
            sv = setAUC.iloc[:,1].mean()
            gl_gv = geneAUC.iloc[:,0].mean()
            gl_sv = setAUC.iloc[:,0].mean()
            f = open('validationscore'+str(clustn)+'.txt', 'a')
            f.write(outname+"\t"+str(gl_gv)+"\t"+str(gv)+"\t"+str(gl_sv)+"\t"+str(sv)+"\n")
            f.close()
        else:
            f = open('validationscore'+str(clustn)+'.txt', 'a')
            f.write(outname+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"+str(0)+"\n")
            f.close()
    except:
        print("Failed: "+print(newlibs[i]))
    shutil.rmtree(predictionFolder, ignore_errors=True)

