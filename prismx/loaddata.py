import urllib.request, json
import hashlib
import pandas as pd
from os import path
import prismx as px
from typing import List
from prismx.utils import loadJSON, getConfig, getDataPath, readGMT

def loadExpression(species: str, overwrite: bool = False) -> str :
    if not path.exists(getDataPath()+species+"_matrix.h5" or overwrite):
        print("Download ARCHS4 expression")
        urllib.request.urlretrieve(getConfig()["ARCH4_S3_BUCKET"]+"/"+species+"_matrix.h5", getDataPath()+species+"_matrix.h5")
    else:
        print("File cached. Reload with loadExpression(\""+species+"\", overwrite=True)")
    return(getDataPath()+species+"_matrix.h5")

def listLibraries():
    return(loadJSON(getConfig()["LIBRARY_LIST_URL"])["library"])

def loadLibrary(library: str, overwrite: bool = False, verbose: bool = False) -> str:
    if not path.exists(getDataPath()+library or overwrite):
        print("Download Enrichr geneset library")
        urllib.request.urlretrieve(getConfig()["LIBRARY_DOWNLOAD_URL"]+library, getDataPath()+library)
    else:
        print("File cached. To reload use loadLibrary(\""+library+"\", overwrite=True) instead.")
    lib, rlib, ugenes = readGMT(getDataPath()+library)
    if verbose:
        print("# genesets: "+str(len(lib))+"\n# unique genes: "+str(len(ugenes)))
    return(getDataPath()+library)

def printLibraries():
    libs = listLibraries()
    for i in range(0, len(libs)):
        print(str(i)+" - "+libs[i])

def getGenes(correlationFolder: str) -> List[str]:
    cc = pd.read_feather(correlationFolder+"/correlation_0.f").set_index("index")
    genes = [x.upper() for x in cc.columns]
    cc = 0
    return(genes)