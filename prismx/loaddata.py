import urllib.request, json
import hashlib
import pandas as pd
from os import path
import prismx as px
from typing import List
from prismx.utils import load_json, get_config, get_data_path, read_gmt

def download_expression(species: str, overwrite: bool = False, verbose: bool = False) -> str :
    if not path.exists(get_data_path()+species+"_matrix.h5" or overwrite):
        if verbose:
            print("Download ARCHS4 expression")
        urllib.request.urlretrieve(get_config()["ARCH4_S3_BUCKET"]+"/"+species+"_matrix.h5", get_data_path()+species+"_matrix.h5")
    else:
        if verbose:
            print("File cached. Reload with download_expression(\""+species+"\", overwrite=True)")
    return(get_data_path()+species+"_matrix.h5")

def list_libraries():
    return(load_json(get_config()["LIBRARY_LIST_URL"])["library"])

def load_library(library: str, overwrite: bool = False, verbose: bool = False) -> str:
    if not path.exists(get_data_path()+library or overwrite):
        if verbose:
            print("Download Enrichr geneset library")
        urllib.request.urlretrieve(get_config()["LIBRARY_DOWNLOAD_URL"]+library, get_data_path()+library)
    else:
        if verbose:
            print("File cached. To reload use load_library(\""+library+"\", overwrite=True) instead.")
    lib, rlib, ugenes = read_gmt(get_data_path()+library)
    if verbose:
        print("# genesets: "+str(len(lib))+"\n# unique genes: "+str(len(ugenes)))
    return(get_data_path()+library)

def print_libraries():
    libs = list_libraries()
    for i in range(0, len(libs)):
        print(str(i)+" - "+libs[i])

def get_genes(workdir: str) -> List[str]:
    cc = pd.read_feather(workdir+"/correlation/correlation_0.f").set_index("index")
    genes = [x.upper() for x in cc.index]
    cc = 0
    return(genes)