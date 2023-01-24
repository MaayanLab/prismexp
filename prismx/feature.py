from typing import Dict, List
import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
import concurrent

from prismx.utils import read_gmt, load_correlation, load_correlation_polars, load_feature
from prismx.loaddata import get_genes


def features(work_dir: str, gmt_file: str, intersect: bool=False, threads: int=2, verbose: bool=False):
    """
    Creates a feature matrix for each correlation matrix in the given directory, using the gene set library in the given gmt file.
    
    Parameters:
    work_dir (str): Path to the directory containing the correlation matrices.
    gmt_file (str): Path to the gmt file containing the gene set library.
    intersect (bool, optional): If True, only includes unique genes present in all gene sets in the feature matrix. Defaults to False.
    threads (int, optional): Number of threads to use for parallel processing. Defaults to 5.
    verbose (bool, optional): If True, prints progress information. Defaults to False.
    """
    # Create the features directory if it does not already exist
    os.makedirs(work_dir+"/features", exist_ok=True)

    # Get a list of the correlation files in the working directory
    correlation_files = os.listdir(work_dir+"/correlation")

    # Load the first correlation matrix
    background_genes = load_correlation(work_dir, 0).columns

    # Initialize an empty list of unique genes
    unique_genes = []

    # Read the gene set library and get the list of unique genes
    library, rev_library, unique_genes = read_gmt(gmt_file, background_genes, verbose=verbose)

    # If the intersect flag is set, get the list of unique genes that are present in all gene sets
    if intersect:
        unique_genes = list(set(sum(library.values(), [])))
        unique_genes = list(set(unique_genes) & set(background_genes))
        unique_genes = [x.encode("UTF-8") for x in unique_genes]
        if verbose:
            print("overlapping genes: "+str(len(unique_genes)))

    # Create a list of integers representing the correlation files to process
    file_indices = list(range(0, len(correlation_files)-1))
    file_indices.append("global")

    # Use a ProcessPoolExecutor to parallelize the function calls
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        if verbose:
            # Submit the function calls and iterate over the completed tasks in the order that they complete
            futures = [executor.submit(get_average_correlation, work_dir, i, library, intersect, unique_genes) for i in file_indices]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                res = future.result()
        else:
            # Submit the function calls and ignore the results
            for i in file_indices:
                executor.submit(get_average_correlation, work_dir, i, library, intersect, unique_genes)

def features_polars(work_dir: str, gmt_file: str, intersect: bool=False, threads: int=2, verbose: bool=False):
    """
    Creates a feature matrix for each correlation matrix in the given directory, using the gene set library in the given gmt file.
    
    Parameters:
    work_dir (str): Path to the directory containing the correlation matrices.
    gmt_file (str): Path to the gmt file containing the gene set library.
    intersect (bool, optional): If True, only includes unique genes present in all gene sets in the feature matrix. Defaults to False.
    threads (int, optional): Number of threads to use for parallel processing. Defaults to 5.
    verbose (bool, optional): If True, prints progress information. Defaults to False.
    """
    # Create the features directory if it does not already exist
    os.makedirs(work_dir+"/features", exist_ok=True)

    # Get a list of the correlation files in the working directory
    correlation_files = os.listdir(work_dir+"/correlation")

    # Load the first correlation matrix
    background_genes = load_correlation(work_dir, 0).columns

    # Initialize an empty list of unique genes
    unique_genes = []

    # Read the gene set library and get the list of unique genes
    library, rev_library, unique_genes = read_gmt(gmt_file, background_genes, verbose=verbose)

    # If the intersect flag is set, get the list of unique genes that are present in all gene sets
    if intersect:
        unique_genes = list(set(sum(library.values(), [])))
        unique_genes = list(set(unique_genes) & set(background_genes))
        unique_genes = [x.encode("UTF-8") for x in unique_genes]
        if verbose:
            print("overlapping genes: "+str(len(unique_genes)))

    # Create a list of integers representing the correlation files to process
    file_indices = list(range(0, len(correlation_files)-1))
    file_indices.append("global")

    for file_idx in file_indices:
        get_average_correlation_polars(work_dir, file_idx, library, intersect, unique_genes)

def get_average_correlation(work_dir: str, i: int, gene_set_library: Dict, intersect: bool=False, unique_genes: List=[]):
    """
    Calculate the average correlation of each gene with a set of genes in a gene set library, and create a feature matrix with genes as rows and gene sets as columns. 
    
    Parameters
    ----------
    work_dir : str
        The working directory where the correlation matrix and resulting feature matrix will be saved.
    i : int
        An integer identifier for the correlation matrix. This will be used as part of the filename when saving the feature matrix.
    gene_set_library : Dict
        A dictionary where the keys are the names of the gene sets and the values are lists of the genes in each gene set.
    intersect : bool, optional
        A flag indicating whether to filter the correlation matrix to only include the unique genes (i.e., the intersection of all gene sets). Defaults to False.
    unique_genes : List, optional
        A list of the unique genes to include in the correlation matrix if the intersect flag is set. This argument is ignored if intersect is False.
    
    Returns
    -------
    int
        Always returns 1.
    """
    correlation_matrix = load_correlation(work_dir, i)
    
    # If intersect flag is set, filter the correlation matrix to only include unique genes
    if intersect:
        correlation_matrix = correlation_matrix.loc[unique_genes, :]

    final_gene_list = correlation_matrix.index

    gene_set_average_correlations = []
    gene_set_names = list(gene_set_library.keys())

    # Calculate the average correlation for each gene set
    for gene_set_name in gene_set_names:
        gene_set_average_correlations.append(np.nanmean(correlation_matrix.loc[:, gene_set_library[gene_set_name]].values, axis=1))

    correlation_matrix = None

    # Stack the average correlations for each gene set vertically to create the feature matrix
    feature_matrix = np.vstack(gene_set_average_correlations).T

    # Replace any NaN values in the feature matrix with 0, should not exist
    np.nan_to_num(feature_matrix, copy=False)

    # Convert the feature matrix to a pandas DataFrame with the gene names as the index and the gene set names as the column names
    feature_matrix = pd.DataFrame(feature_matrix, columns=gene_set_names, index=final_gene_list).reset_index()

    # Convert the column names to strings
    feature_matrix.columns = feature_matrix.columns.astype(str)

    # Write the feature matrix to disk as a feather file
    feature_matrix.to_feather(work_dir+"/features/features_"+str(i)+".f")
    feature_matrix = None

    return 1


def get_average_correlation_polars(work_dir: str, i: int, gene_set_library: Dict, intersect: bool=False, unique_genes: List=[]):
    """
    Calculate the average correlation of each gene with a set of genes in a gene set library, and create a feature matrix with genes as rows and gene sets as columns. 
    
    Parameters
    ----------
    work_dir : str
        The working directory where the correlation matrix and resulting feature matrix will be saved.
    i : int
        An integer identifier for the correlation matrix. This will be used as part of the filename when saving the feature matrix.
    gene_set_library : Dict
        A dictionary where the keys are the names of the gene sets and the values are lists of the genes in each gene set.
    intersect : bool, optional
        A flag indicating whether to filter the correlation matrix to only include the unique genes (i.e., the intersection of all gene sets). Defaults to False.
    unique_genes : List, optional
        A list of the unique genes to include in the correlation matrix if the intersect flag is set. This argument is ignored if intersect is False.
    
    Returns
    -------
    int
        Always returns 1.
    """
    correlation_matrix = load_correlation_polars(work_dir, i)
    
    # If intersect flag is set, filter the correlation matrix to only include unique genes
    if intersect:
        correlation_matrix = correlation_matrix.loc[unique_genes, :]

    final_gene_list = correlation_matrix.index

    gene_set_average_correlations = []
    gene_set_names = list(gene_set_library.keys())

    # Calculate the average correlation for each gene set
    for gene_set_name in gene_set_names:
        gene_set_average_correlations.append(np.nanmean(correlation_matrix.loc[:, gene_set_library[gene_set_name]].values, axis=1))

    correlation_matrix = None

    # Stack the average correlations for each gene set vertically to create the feature matrix
    feature_matrix = np.vstack(gene_set_average_correlations).T

    # Replace any NaN values in the feature matrix with 0, should not exist
    np.nan_to_num(feature_matrix, copy=False)

    # Convert the feature matrix to a pandas DataFrame with the gene names as the index and the gene set names as the column names
    feature_matrix = pd.DataFrame(feature_matrix, columns=gene_set_names, index=final_gene_list).reset_index()

    # Convert the column names to strings
    feature_matrix.columns = feature_matrix.columns.astype(str)

    # Write the feature matrix to disk as a feather file
    feature_matrix.to_feather(work_dir+"/features/features_"+str(i)+".f")
    feature_matrix = None

    return 1

def load_features(work_dir: str, verbose: bool=False): 
    features = []
    feature_files = os.listdir(work_dir+"/features")
    lk = list(range(0, len(feature_files)-1))
    lk.append("global")
    for i in lk:
        feature = load_feature(work_dir,i)
        features.append(feature)
        if verbose:
            print("features_"+str(i)+".f")
    return(features)

def load_features_range(workdir: str, range_from: int, range_to: int, verbose: bool=False): 
    features = []
    feature_files = os.listdir(workdir+"/features")
    lk = list(range(0, len(feature_files)-1))
    lk.append("global")
    for i in lk:
        feature = load_feature(workdir, i)
        features.append(feature.iloc[:, range_from:range_to].copy())
        feature = 0
        if verbose:
            print("features_"+str(i)+".f")
    return(features)
