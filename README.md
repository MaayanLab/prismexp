<img align="left" width="200" src="https://mssm-prismx.s3.amazonaws.com/images/prismxsmall.png">

# PrismEXP

## Package for gene function predictions by unsupervised gene expression partitioning

Gene co-expression is a commonly used feature in many machine learning applications. The elucidation of gene function frequently relies on the use of correlation structures, and the performance of the predictions relies on the chosen gene expression data. In some applications, correlations derived from tissue-specific gene expression outperform correlations derived from global gene expression. However, the identification of the optimal tissue may not always be trivial, and the constraint of a single tissue might be too limiting in some circumstances. To address this problem, we introduce and validate a new statistical approach, Automated Sharding of Massive Co-expression RNA-seq Data (PrismEXP), for accurate gene function prediction. We apply PrismEXP on ARCHS4 gene expression to predict a wide variety of gene properties, such as pathway memberships, phenotype associations, and protein-protein interactions. PrismEXP outperforms single correlation matrix approaches on all tested domains. The proposed method can enhance existing machine learning methods using gene correlation information and will require only minor adjustments to existing algorithms.

This Python3 package allows the generation of correlation matrices needed for the prediction of gene function from GMT files. The memory requirements depend on the number of genes used and the number of gene expression profiles.

Default settings and ARCHS4 mouse gene expression should require less than 8GB of memory. The file formats used are hdf5 and feather. Gene expression has to be provided in H5 format. Gene expression should be stored as a matrix under "data/expression", gene symbols under "meta/genes", and sample identifieres under "meta/Sample_geo_accession"

Precomputed PrismExp predictions for popular Enrichr gene set libraries can be accessed here: https://maayanlab.cloud/prismexp<br>
The PrismExp Appyter for all Enrichr libraries can be accessed here: https://appyters.maayanlab.cloud/PrismEXP/

---
**NOTE**
javascript:downloadFile('https://s3.dev.maayanlab.cloud/archs4/archs4_gene_human_v2.1.2.h5','human_matrix_v2.1.2.h5','2.1.2')
PrismEXP requires a large gene expression repository. The code expects gene expression as gene counts. Data compatible with PrismX can be downloaded from the ARCHS4 website.<br><br>
Mouse Data (717,966 samples): https://s3.dev.maayanlab.cloud/archs4/archs4_gene_mouse_v2.1.2.h5<br>
Human Data (620,825 samples): https://s3.dev.maayanlab.cloud/archs4/archs4_gene_human_v2.1.2.h5

---


## Installation

Install the python package directly from Github using PIP.

```
$ pip install git+https://github.com/MaayanLab/prismexp.git
```

## Usage

### Create gene correlation matrices
Creating gene-gene correlation matrices requires 4 steps
1. Download ARCHS4 gene expression: [https://mssm-seq-matrix.s3.amazonaws.com/mouse_matrix.h5](https://s3.dev.maayanlab.cloud/archs4/archs4_gene_human_v2.1.2.h5) (there is test data included in the package)
2. filter genes with low expression
3. partition gene expression profiles into a set of distinct clusters
4. calculate gene-gene correlation within each cluster

### Create gene function predictions
Creating the predictions requires the gene-gene correlation matrices as a prerequisite
1. Provide a GMT file. (Samples of GMT file can be found at: https://amp.pharm.mssm.edu/Enrichr/#stats)
    * each line of a GMT is tab separated and starts with a gene set name followed by a description, followed by gene symbols. Example: potassium ion import (GO:0010107) \t description \t SLC12A3 \t KCNJ5 \t SLC12A4 \t KCNJ6 \t ...
2. Create gene expression cluster wise predictions
3. Assemble cluster based predictions
4. Apply trained PRISMX machine learning model

## Code example

The following example will download the ARCHS4 gene expression and build 50 gene expression clusters. This process will, depending on the used hardware, take considerable amount of time. It also requires about 1GB of diskspace per gene expression cluster. Additional to the 50 gene-gene matrices the algorithm will also compute a correlation matrix across clusters. Memory consumption will depend on clustering, but should stay below 8GB.

### Python3

### I) Compute correlation matrices

The choice of clusters will impact the overall quality of gene function predictions. The predictions improve proportional to the log of the number of clusters. Adding more clusters will increase the runtime of the algorithm. If possible we recommend 200-300 clusters. Beyond 300 clusters improvements are marginal.

```python
import urllib.request
import prismx as px
import os

urllib.request.urlretrieve("https://s3.dev.maayanlab.cloud/archs4/archs4_gene_human_v2.1.2.h5", "human_matrix.h5")

correlationFolder = "correlation_folder"
clusterNumber = 200

os.mkdir(correlationFolder)
px.createCorrelationMatrices("mouse_matrix.h5", correlationFolder, clusterCount=clusterNumber, sampleCount=5000, verbose=True)
```

### II) Calculate average correlation of genes to gene sets for given gene set library

```python
import prismx as px

# reuse correlation matrices from first step
correlationFolder = "correlation_folder"
predictionFolder = "prediction_folder"
libs = px.list_libraries()

# select GO: Biological Processes
gmt_file = px.load_library(libs[110])

px.correlation_scores(gmt_file, correlationFolder, predictionFolder, verbose=True)
```

### III) Train model on GO: Biological Processes gene set library

```python
import prismx as px
import pickle

# reuse matrices from step I and step II
correlationFolder = "correlation_folder"
predictionFolder = "prediction_folder"
libs = px.list_libraries()

# select GO: Biological Processes (same as last step)
gmt_file = px.load_library(libs[110])

model = px.trainModel(predictionFolder, correlationFolder, gmt_file, training_size=300000, test_train_split=0.1, sample_positive=40000, sample_negative=200000, random_state=42, verbose=True)
pickle.dump(model, open("gobp_model.pkl", 'wb'))
```

Once the model is trained it can be applied on any gene set library of choice. Models trained with GO: BP were tested on all gene set libraries in Enrichr and show on average for all gene set libraries.

### IV) Predict gene functions

```python
import prismx as px

# reuse matrices from step I and step II
correlationFolder = "correlation_folder"
predictionFolder = "prediction_folder"
libs = px.list_libraries()

# choose a gene set library from Enrichr
i = 1
outname = libs[i]
gmt_file = px.load_library(libs[i])

outfolder = "prismxresult"

px.predict_gmt("gobp_model.pkl", gmt_file, correlationFolder, predictionFolder, outfolder, outname, step_size=200, intersect=False, verbose=True)
```

## Bridge Gene Set Enrichment Analysis (bridgeGSEA)

Use PrismEXP gene set predictions in enrichment analysis to identify novel genes in enriched pathways and biological processes.

```python
import prismx as px
import prismx.gsea as pxgsea
import blitzgsea as blitz
import urllib.request

url = "https://github.com/MaayanLab/blitzgsea/raw/main/testing/ageing_muscle_gtex.tsv"
urllib.request.urlretrieve(url, "ageing_muscle_gtex.tsv")

# read signature as pandas dataframe
signature = pd.read_csv("ageing_muscle_gtex.tsv")

# use enrichr submodule to retrieve gene set library
library = blitz.enrichr.get_library("GO_Biological_Process_2021")

result = px.bridgegsea.bridge_gsea(signature, library, predictions)
```
