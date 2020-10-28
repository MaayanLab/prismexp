import urllib.request
import prismx as px
import pickle
import os
import time
import matplotlib

clusterCount = 50

correlationFolder = "correlation_"+str(clusterCount)+"_folder"
predictionFolder = "prediction_"+str(clusterCount)+"_folder"
outfolder = "prismxresult_"+str(clusterCount)
#px.createCorrelationMatrices("mouse_matrix.h5", correlationFolder, clusterCount=clusterCount, sampleCount=5000, correlationSampleCount=5000, verbose=True)

genesetlibs = ["ChEA_2016", "KEA_2013", "GWAS_Catalog_2019", "huMAP", "GO_Biological_Process_2018", "MGI_Mammalian_Phenotype_Level_4_2019"]
genesetlibs.sort()

for lib in genesetlibs[0:2]:
    gmtFile = px.loadLibrary(lib)
    px.correlationScores(gmtFile, correlationFolder, predictionFolder, verbose=True)
    model = px.trainModel(predictionFolder, correlationFolder, gmtFile, trainingSize=300000, testTrainSplit=0.1, samplePositive=40000, sampleNegative=200000, randomState=42, verbose=True)
    pickle.dump(model, open(lib+"_model_"+str(clusterCount)+".pkl", 'wb'))


outfolder = "prismxresult_"+str(clusterCount)

os.makedirs("testdata", exist_ok=True)
for lib in genesetlibs[0:2]:
    for lib2 in genesetlibs[0:6]:
        outname = lib2
        gmtFile = px.loadLibrary(lib2)
        px.predictGMT(lib+"_model_"+str(clusterCount)+".pkl", gmtFile, correlationFolder, predictionFolder, outfolder, outname, stepSize=200, intersect=True, verbose=True)
        # benchmark the prediction quality
        geneAUC, setAUC = px.benchmarkGMTfast(gmtFile, correlationFolder, predictionFolder, outfolder+"/"+outname+".f", intersect=True, verbose=True)
        geneAUC = geneAUC.reset_index()
        setAUC = setAUC.reset_index()
        geneAUC.to_feather("testdata/auc_gene_"+lib+"_"+lib2+".f")
        setAUC.to_feather("testdata/auc_set_"+lib+"_"+lib2+".f")




modelq_set_g = pd.DataFrame()
for lib in genesetlibs[0:6]:
    lq = list()
    for lib2 in genesetlibs[0:6]:
        aauc = pd.read_feather("testdata/modelq/auc_set_"+lib+"_"+lib2+".f").mean()[0]
        lq.append(aauc)
    modelq_set_g[lib] = lq

modelq_set_g.index = genesetlibs


modelq_set = pd.DataFrame()
for lib in genesetlibs[0:6]:
    lq = list()
    for lib2 in genesetlibs[0:6]:
        aauc = pd.read_feather("testdata/modelq/auc_set_"+lib+"_"+lib2+".f").mean()[1]
        lq.append(aauc)
    modelq_set[lib] = lq

modelq_set.index = genesetlibs

modelq_gene_g = pd.DataFrame()
for lib in genesetlibs[0:6]:
    lq = list()
    for lib2 in genesetlibs[0:6]:
        aauc = pd.read_feather("testdata/modelq/auc_gene_"+lib+"_"+lib2+".f").mean()[0]
        lq.append(aauc)
    modelq_gene_g[lib] = lq

modelq_gene_g.index = genesetlibs


modelq_gene = pd.DataFrame()
for lib in genesetlibs[0:6]:
    lq = list()
    for lib2 in genesetlibs[0:6]:
        aauc = pd.read_feather("testdata/modelq/auc_gene_"+lib+"_"+lib2+".f").mean()[1]
        lq.append(aauc)
    modelq_gene[lib] = lq

modelq_gene.index = genesetlibs

np.fill_diagonal(modelq_gene.values, np.NaN)

labels = ["ChEA", "GO: Biological Process", "GWAS Catalog", "KEA", "MGI Mammalian Phenotype", "huMAP"]

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if not ax:
        ax = plt.gca()
    plt.rcParams["axes.grid"] = False
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=16)
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts


plt.close()
plt.rcParams["axes.grid"] = False
fig, ax = plt.subplots()
im, cbar = heatmap(modelq_gene, labels, labels, ax=ax,
                   cmap="magma_r", cbarlabel="AUC")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
fig.tight_layout()
plt.show()
