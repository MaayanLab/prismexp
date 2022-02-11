import pandas as pd
import feather
import numpy as np
import os
import h5py as h5
import requests
import json

from matplotlib import pyplot as plt
import seaborn
seaborn.set(style='ticks')
import scipy.stats as stats

burl = "https://maayanlab.cloud/prismexpapi/"

def libraries():
    url = burl+"libraries"
    resp = requests.post(url)
    return resp.json()

def genes(library):
    url = burl+"genes"
    payload={"library": library}
    resp = requests.post(url, json=payload)
    return resp.json()

def nes(rank_vector, geneset):
    gs = set(geneset)
    hits = [i for i,x in enumerate(rank_vector.index) if x in gs]
    
    hit_indicator = np.zeros(rank_vector.shape[0])
    hit_indicator[hits] = 1
    
    no_hit_indicator = 1 - hit_indicator
    
    number_hits = len(hits)
    number_miss = rank_vector.shape[0] - number_hits
    
    sum_hit_scores = np.sum(rank_vector.iloc[hits]) 
    
    norm_hit =  1.0/sum_hit_scores
    norm_no_hit = 1.0/number_miss
    
    running_sum = np.cumsum(hit_indicator * rank_vector * norm_hit - no_hit_indicator * norm_no_hit)
    max_ES, min_ES =  running_sum.max(), running_sum.min()
    
    es_vec = np.where(np.abs(max_ES) > np.abs(min_ES), max_ES, min_ES)
    
    return running_sum, es_vec

def load_prediction(library, geneset):
    url = burl+"predictions"
    payload = {"library": library, "id": geneset.upper()}
    res = requests.post(url, json=payload)
    res = res.json()
    df = pd.DataFrame(data=np.array(res["values"]))
    df.index = res["genes"]
    return df

def gsea(signature, library, geneset, glib):
    prediction = load_prediction(library, geneset)
    plt, top_pred =  plot_gsea(signature, glib[geneset.upper()], prediction, title=geneset)
    return plt, top_pred

def plot_gsea(signature, geneset, prediction, title="", pred_count=100, max_highlight = 30):
    
    rank_vec = signature
    rank_vec = rank_vec.sort_values(ascending=False)
    prediction = prediction.sort_values(0, ascending=False)
    
    hits = [i for i,x in enumerate(rank_vec.index) if x in geneset]
    
    pred_genes = set(prediction.index[0:pred_count])
    pred_genes = pred_genes.intersection(rank_vec.index).difference(geneset)
    pred_hits = [i for i,x in enumerate(rank_vec.index) if x in pred_genes]
    
    combined_hits = set(list(list(geneset)+list(pred_genes)))
    
    running_sum_orig, es_orig = nes(rank_vec, geneset)
    running_sum, es = nes(rank_vec, combined_hits)
    running_sum = list(running_sum)
    
    fig = plt.figure(figsize=(11,5))
    ax = fig.add_gridspec(12, 12, wspace=0, hspace=0)
    ax1 = fig.add_subplot(ax[0:7, 0:8])
    ax1.plot(list(running_sum_orig), color=(0.5,0.5,0.5), lw=2)
    ax1.plot(running_sum, color=(0,1,0), lw=3)
    plt.xlim([0, len(running_sum)])

    nn = np.where(np.abs(running_sum)==np.max(np.abs(running_sum)))[0][0]
    ax1.vlines(x=nn, ymin=np.min(running_sum), ymax=np.max(running_sum),linestyle = ':', color="red")
    
    if es > 0:
        ax1.text(len(running_sum)/30, 0, "ES="+"{:.3f}".format(running_sum[nn]), size=20, bbox={'facecolor':'white','alpha':0.8,'edgecolor':'none','pad':1}, ha='left', va='bottom', zorder=100)
    else:
        ax1.text(len(running_sum)/30, 0, "ES="+"{:.3f}".format(running_sum[nn]), size=20, bbox={'facecolor':'white','alpha':0.8,'edgecolor':'none','pad':1}, ha='left', va='top', zorder=100)

    ax1.grid(True, which='both')
    ax1.set(xticks=[])
    seaborn.despine(ax=ax1, offset=0)
    
    if title != "":
        plt.title(title)
        
    plt.ylabel("Enrichment Score (ES)")

    ax1 = fig.add_subplot(ax[7:9, 0:8])

    ax1.vlines(x=pred_hits, ymin=-1, ymax=0, color=(1,0,1,1), lw=0.5)
    ax1.vlines(x=hits, ymin=0, ymax=1, color=(0,0,0,1), lw=0.5)
    plt.xlim([0, len(running_sum)])
    plt.ylim([-1, 1])

    ax1.set(yticks=[])
    ax1.set(xticks=[])

    ax1 = fig.add_subplot(ax[9:, 0:8])

    x = np.arange(0.0, len(rank_vec), 20).astype("int")
    x = np.append(x, len(rank_vec)-1)

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
    plt.xlabel("Rank in Ordered Dataset")
    plt.ylabel("Ranked list metric")

    ax1 = fig.add_subplot(ax[:, 9:])

    if es > 0:
        pred_le = np.where(pred_hits < nn)[0][0:np.min([max_highlight, len(pred_hits)])]
        bars = rank_vec.index[pred_hits][pred_le][::-1]
    else:
        pred_le = np.where(pred_hits > nn)[0][0:np.min([max_highlight, len(pred_hits)])]
        bars = rank_vec.index[pred_hits][pred_le]

    y_pos = np.arange(len(bars))
    
    plt.yticks(y_pos, bars)

    bar_height = 0.4
    bar_width = 0.4
    
    v = ((prediction.loc[bars]/np.max(prediction))*np.max(np.abs(rank_vec.loc[bars])))[0].values
    print(np.max(prediction.loc[bars]))
    b1 = ax1.barh(y_pos, v, height=bar_height, color="dodgerblue")
    
    b2 = ax1.barh(y_pos + bar_width, list(rank_vec.loc[bars]), height=bar_height, color="black")

    # Fix the x-axes.
    ax1.set_yticks(y_pos + bar_width / 2)
    ax1.set_yticklabels(bars)
    plt.xlabel("Predicted leading edge rank metric")
    sm = np.max(np.abs(list(rank_vec.loc[bars])))
    #plt.xlim([np.min(list(rank_vec.loc[bars])), np.max(list(rank_vec.loc[bars]))])
    if np.min(list(rank_vec.loc[bars])) > 0:
        plt.xlim([0, sm])
    else:
        plt.xlim([-sm, sm])
    plt.box(False)

    ax2 = ax1.twiny()
    #ax2.set_xticks(np.linspace(0,sm,6))
    if np.min(list(rank_vec.loc[bars])) > 0:
        ax2.set_xticks(np.linspace(0,sm,4))
    else:
        ax2.set_xticks(np.linspace(sm,2*sm,4))
    
    ax2.set_xticklabels(np.around(np.linspace(0,prediction.loc[bars].max()[0],4),1))
    plt.xlabel("PrismExp Association z-Score", c="dodgerblue")

    plt.box(False)
    
    print(bars)
    
    ll = list()
    ll.append(list(bars))
    ll.append(list(rank_vec.loc[bars]))
    ll.append(list(prediction.loc[bars][0]))
    ll.append(stats.norm.sf(np.absolute(np.array(prediction.loc[bars][0]))))
    top_pred = pd.DataFrame(ll).T
    top_pred = top_pred.set_index(0)
    top_pred.columns = ["metric","prediction", "p-value"]
    top_pred["pxscore"] = top_pred["metric"]*top_pred["prediction"]
    top_pred = top_pred.sort_values("pxscore", ascending=False)
    top_pred.index.name = None
    
    return plt, top_pred