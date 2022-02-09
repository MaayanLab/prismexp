import os
import sys
import time
import shutil
#sys.path.append('C:/prismx/')
import prismx as px

libs = px.list_libraries()
clustn = 26

f = open("validationscore"+str(clustn)+".txt", 'r')
libraries = [x.split("\t")[0] for x in f.readlines()]
newlibs = list(set(libs).difference(set(libraries)))

for i in range(0, len(newlibs)):
    try:
        print(newlibs[i])
        gmt_file = px.load_library(newlibs[i])
        print("loaded")
        g1, g2, g3 = px.read_gmt(gmt_file)
        # set output configuration
        outname = newlibs[i]
        correlationFolder = "correlation_"+str(clustn)+"_folder"
        predictionFolder = "prediction_"+str(clustn)
        outfolder = "prismxresult_"+str(clustn)
        if len(g1) < 14000:
            # calculate PrismX predictions with pretrained model
            px.predict_gmt("gobp_model_"+str(clustn)+".pkl", gmt_file, correlationFolder, predictionFolder, outfolder, outname, step_size=200, intersect=True, verbose=True)
            # benchmark the prediction quality
            geneAUC, setAUC = px.benchmarkGMTfast(gmt_file, correlationFolder, predictionFolder, outfolder+"/"+outname+".f", intersect=True, verbose=True)
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

