import pandas as pd
import feather
import numpy as np
import os
import requests
import json

from matplotlib import pyplot as plt
import gseapy
import scipy.stats as stats

def nesold(rank_vector, gene_set):
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

def nes(signature, gene_set):
    rank_vector = signature.sort_values(1, ascending=False).set_index(0)
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
    running_sum = np.cumsum(hit_indicator * np.abs(rank_vector[1]) * float(norm_hit) - no_hit_indicator * float(norm_no_hit))
    nn = np.where(np.abs(running_sum)==np.max(np.abs(running_sum)))[0][0]
    es = running_sum[nn]
    return running_sum, es

def bridge_genesets(signature, pred, top_sets, library, pred_gene_number=50):
    bridge_library = {}
    for i in list(top_sets):
        gs = list(set(library[i.upper()]))
        bridge_library[i] = gs
        diff_genes = list(set(signature[0]).difference(gs).intersection(set(pred.index)))
        pred_genes = set(pred[i.upper()].loc[diff_genes].sort_values(ascending=False).iloc[0:pred_gene_number].index)
        bridge_library[i].extend(pred_genes)
        bridge_library[i] = sorted(list(set(bridge_library[i])))
    return bridge_library

def filter_ledge(combined_scores):
    filtered_ledge = []
    for i in range(combined_scores.shape[0]):
        filtered_ledge.append(sorted(list(set(combined_scores.iloc[i,10].split(";")).difference(set(combined_scores.iloc[i,7].split(";"))))))
    return filtered_ledge

def bridge_gsea(signature, library, predictions, permutations=100, pred_gene_number=50, processes=1):
    signature.index = signature[0]
    signature = signature.sort_values(1, ascending=False)
    signature = signature[~signature.index.duplicated(keep='first')]
    pre_res = gseapy.prerank(rnk=signature, gene_sets=library, processes=processes, permutation_num=permutations, outdir='test/prerank_report_kegg', format='png', seed=1)
    gsea_res = pre_res.res2d
    bridge_library = bridge_genesets(signature, predictions, gsea_res.index, library, pred_gene_number=pred_gene_number)
    pre_res = gseapy.prerank(rnk=signature, gene_sets=bridge_library, processes=processes, permutation_num=permutations, outdir='test/prerank_report_kegg', format='png', seed=1)
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

def plot_gsea(signature, geneset, library, prediction, pred_gene_number=50, max_highlight=20):
    signature.index = signature[0]
    signature = signature.sort_values(1, ascending=False)
    signature = signature[~signature.index.duplicated(keep='first')]
    gs = set(library[geneset])
    hits = [i for i,x in enumerate(signature[0]) if x in gs]
    diff_genes = list(set(signature[0]).difference(gs).intersection(set(prediction.index)))
    pred_genes = set(prediction[geneset.upper()].loc[diff_genes].sort_values(ascending=False).iloc[0:pred_gene_number].index)
    pred_hits = [i for i,x in enumerate(signature[0]) if x in pred_genes]
    combined_hits = set(list(list(gs)+list(pred_genes)))
    running_sum_orig, es_orig = nes(signature, gs)
    running_sum, es = nes(signature, combined_hits)
    running_sum = list(running_sum)
    fig = plt.figure(figsize=(11,5))
    ax = fig.add_gridspec(12, 12, wspace=0, hspace=0)
    ax1 = fig.add_subplot(ax[0:7, 0:8])
    ax1.plot(list(running_sum_orig), color=(0.5,0.5,0.5), lw=2)
    ax1.plot(running_sum, color=(0,1,0), lw=3)
    plt.xlim([0, len(running_sum)])
    # --------------------
    nn = np.where(np.abs(running_sum)==np.max(np.abs(running_sum)))[0][0]
    ax1.vlines(x=nn, ymin=np.min(running_sum), ymax=np.max(running_sum),linestyle = ':', color="red")
    if es > 0:
        ax1.text(len(running_sum)/30, 0, "ES="+"{:.3f}".format(running_sum[nn]), size=20, bbox={'facecolor':'white','alpha':0.8,'edgecolor':'none','pad':1}, ha='left', va='bottom', zorder=100)
    else:
        ax1.text(len(running_sum)/30, 0, "ES="+"{:.3f}".format(running_sum[nn]), size=20, bbox={'facecolor':'white','alpha':0.8,'edgecolor':'none','pad':1}, ha='left', va='top', zorder=100)
    ax1.grid(True, which='both')
    ax1.set(xticks=[])
    #seaborn.despine(ax=ax1, offset=0)
    plt.title(geneset)
    plt.ylabel("Enrichment Score (ES)", fontsize=16)
    #-----------------------
    ax1 = fig.add_subplot(ax[7:9, 0:8])
    ax1.vlines(x=pred_hits, ymin=-1, ymax=0, color=(1,0,1,1), lw=0.5)
    ax1.vlines(x=hits, ymin=0, ymax=1, color=(0,0,0,1), lw=0.5)
    plt.xlim([0, len(running_sum)])
    plt.ylim([-1, 1])
    ax1.set(yticks=[])
    ax1.set(xticks=[])
    ax1 = fig.add_subplot(ax[9:, 0:8])
    rank_vec = signature[1]
    x = np.arange(0.0, len(rank_vec), 20).astype("int")
    x = np.append(x, signature.shape[0]-1)
    ax1.fill_between(x, np.array(rank_vec)[x], color="lightgrey")
    ax1.plot(x, np.array(rank_vec)[x], color=(0.2,0.2,0.2), lw=1)
    ax1.hlines(y=0, xmin=0, xmax=len(rank_vec), color="black", zorder=100, lw=0.6)
    plt.xlim([0, len(running_sum)])
    plt.ylim([np.min(rank_vec), np.max(rank_vec)])
    minabs = np.min(np.abs(rank_vec))
    zero_cross = int(np.where(np.abs(rank_vec)==minabs)[0][0])
    ax1.vlines(x=zero_cross, ymin=np.min(rank_vec), ymax=np.max(rank_vec),linestyle = ':',)
    ax1.text(zero_cross, np.max(rank_vec)/3, "Zero crosses at "+str(zero_cross), bbox={'facecolor':'white','alpha':0.5,'edgecolor':'none','pad':1}, ha='center', va='center')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.xlabel("Rank in Ordered Dataset", fontsize=16)
    plt.ylabel("Ranked list metric", fontsize=16)
    ax1 = fig.add_subplot(ax[:, 9:])
    if es > 0:
        pred_le = np.where(pred_hits < nn)[0][0:np.min([max_highlight, len(pred_hits)])]
        bars = rank_vec.index[pred_hits][pred_le][::-1]
    else:
        pred_le = np.where(pred_hits > nn)[0][0:np.min([max_highlight, len(pred_hits)])]
        bars = rank_vec.index[pred_hits][pred_le]
    y_pos = np.arange(len(bars))
    #ax1.barh(y_pos, height, color="black")
    plt.yticks(y_pos, bars)
    bar_width = 0.4
    pp = prediction[geneset.upper()].sort_values(ascending=False)
    pp = (pp - pp.mean())/pp.std(ddof=0)
    # Fix the x-axes.
    ax1.set_yticks(y_pos + bar_width / 2)
    ax1.set_yticklabels(bars)
    plt.xlabel("Predicted leading edge rank metric", fontsize=16)
    sm = np.max(np.abs(list(rank_vec.loc[bars])))
    #plt.xlim([np.min(list(rank_vec.loc[bars])), np.max(list(rank_vec.loc[bars]))])
    plt.xlim([-sm, sm])
    plt.box(False)
    ax2 = ax1.twiny()
    #ax2.set_xticks(np.linspace(0,sm,6))
    ax2.set_xticks(np.linspace(sm,2*sm,4))
    ax2.set_xticklabels(np.around(np.linspace(0,np.max(pp),4),1))
    plt.xlabel("PrismExp Association z-Score", c="dodgerblue", fontsize=16)
    plt.box(False)
    return fig