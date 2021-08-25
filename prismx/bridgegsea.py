import pandas as pd
import feather
import numpy as np
import os
import requests
import json

from matplotlib import pyplot as plt
import gseapy
import scipy.stats as stats

def nes(rank_vector, gene_set):
    rank_vector = rank_vector.sort_values(0, ascending=False)
    gs = set(gene_set)
    hits = [i for i,x in enumerate(rank_vector.index) if x in gs]
    hit_indicator = np.zeros(rank_vector.shape[0])
    hit_indicator[hits] = 1
    no_hit_indicator = 1 - hit_indicator
    number_hits = len(hits)
    number_miss = rank_vector.shape[0] - number_hits
    sum_hit_scores = np.sum(np.abs(rank_vector.iloc[hits]))
    norm_hit =  1.0/sum_hit_scores
    norm_no_hit = 1.0/number_miss
    running_sum = np.cumsum(hit_indicator * np.abs(rank_vector) * norm_hit - no_hit_indicator * norm_no_hit)
    nn = np.where(np.abs(running_sum)==np.max(np.abs(running_sum)))[0][0]
    es = running_sum[nn]
    return running_sum, es

def bridge_genesets(pred, top_sets, library, pred_gene_number=50):
    bridge_library = {}
    for i in list(top_sets):
        gs = list(set(library[i.upper()]))
        bridge_library[i] = gs
        pred_genes = set(pred[i.upper()].sort_values(ascending=False).iloc[0:pred_gene_number].index)
        bridge_library[i].extend(pred_genes)
        bridge_library[i] = sorted(list(set(bridge_library[i])))
    return bridge_library

def filter_ledge(combined_scores):
    filtered_ledge = []
    for i in range(combined_scores.shape[0]):
        filtered_ledge.append(sorted(list(set(combined_scores.iloc[i,10].split(";")).difference(set(combined_scores.iloc[i,7].split(";"))))))
    return filtered_ledge

def bridge_gsea(signature, library, predictions):
    pre_res = gseapy.prerank(rnk=signature, gene_sets=library, processes=8, permutation_num=1000, outdir='test/prerank_report_kegg', format='png', seed=1)
    gsea_res = pre_res.res2d
    bridge_library = bridge_genesets(predictions, gsea_res.index, library, pred_gene_number=50)
    pre_res = gseapy.prerank(rnk=tt, gene_sets=bridge_library, processes=8, permutation_num=1000, outdir='test/prerank_report_kegg', format='png', seed=1)
    bridge_gsea_res = pre_res.res2d
    combined_enrichment = pd.concat([gsea_res, bridge_gsea_res], join="inner", axis=1)
    coln = np.array(combined_enrichment.columns)
    coln[8:] = ["bridged_"+x for x in coln[8:]]                           
    combined_enrichment.columns = coln
    combined_enrichment = combined_enrichment.sort_values(by="bridged_es",ascending=False)
    pred_ledges = []
    for i in range(combined_enrichment.shape[0]):
        ledge_genes = combined_enrichment.loc[:,"ledge_genes"].iloc[i].split(";")
        bridge_ledge_genes = combined_enrichment.loc[:,"bridged_ledge_genes"].iloc[i].split(";")
        predicted_ledge = sorted(list(set(bridge_ledge_genes).difference(set(ledge_genes))))
        predicted_ledge = ";".join(predicted_ledge)
        pred_ledges.append(predicted_ledge)
    combined_enrichment["predicted_ledge"] = np.array(pred_ledges)
    return combined_enrichment

def plot_enrichment(enrichment):
    f, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(enrichment.iloc[:,1], enrichment.iloc[:,9])
    ax.plot([enrichment.iloc[:,[1,9]].min().min(), enrichment.iloc[:,[1,9]].max().max()], [enrichment.iloc[:,[1,9]].min().min(), enrichment.iloc[:,[1,9]].max().max()], ls="--", c=".3")
    ax.set_xlabel('NES', fontsize=20)
    ax.set_ylabel('bridged NES', fontsize=20)
    return f