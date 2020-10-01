<img align="left" width="200" src="https://mssm-prismx.s3.amazonaws.com/images/prismxsmall.png">

# PrismX

## Package for gene function predictions by unsupervised gene expression partitioning

Gene co-expression is a commonly used feature in many machine learning applications. The elucidation of gene function frequently relies on the use of correlation structures, and the performance of the predictions relies on the chosen gene expression data. In some applications, correlations derived from tissue-specific gene expression outperform correlations derived from global gene expression. However, the identification of the optimal tissue may not always be trivial, and the constraint of a single tissue might be too limiting in some circumstances. To address this problem, we introduce and validate a new statistical approach, Automated Sharding of Massive Co-expression RNA-seq Data (find fun acronym), for accurate gene function prediction. We apply FunAcronym on ARCHS4 gene expression to predict a wide variety of gene properties, such as pathway memberships, phenotype associations, and protein-protein interactions. FunAcronym outperforms single correlation matrix approaches on all tested domains. The proposed method can enhance existing machine learning methods using gene correlation information and will require only minor adjustments to existing algorithms.

This Python3 package allows the generation of correlation matrices needed for the prediction of gene function from GMT files. The memory requirements depend on the number of genes used and the number of gene expression profiles.

Default settings and ARCHS4 mouse gene expression should require less than 16GB of memory. The file formats used are hdf5 and feather. Gene expression has to be provided in H5 format. Gene expression should be stored as a matrix under "data/expression", gene symbols under "meta/genes", and sample identifieres under "meta/Sample_geo_accession"

---
**NOTE**

PrismX requires a large gene expression repository. The code expects gene expression as gene counts. Data compatible with PrismX can be downloaded from the ARCHS4 website.<br><br>
Mouse Data (284907 samples): https://mssm-seq-matrix.s3.amazonaws.com/mouse_matrix.h5<br>
Human Data (238522 samples): https://mssm-seq-matrix.s3.amazonaws.com/human_matrix.h5

---

## Usage

### Create gene correlation matrices
Creating gene-gene correlation matrices requires 4 steps
1. Download ARCHS4 gene expression: https://mssm-seq-matrix.s3.amazonaws.com/mouse_matrix.h5 (there is test data included in the package)
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

The following example will download the ARCHS4 gene expression and build 50 gene expression clusters. This process will, depending on the used hardware, take considerable amount of time. It also requires about 4GB of diskspace per gene expression cluster. (50*4 ~ 200GB). Additional to the 50 gene-gene matrices the algorithm will also compute a coorelation matrix across clusters. Memory consumption will depend on clustering, but should stay below 16GB.

### Python3

### I) Compute correlation matrices

```python
import urllib.request
import prismx as px
import os

urllib.request.urlretrieve("https://mssm-seq-matrix.s3.amazonaws.com/mouse_matrix.h5", "mouse_matrix.h5")

correlationFolder = "correlation_folder"
clusterNumber = 50

os.mkdir(outFolder)
avg_cor = px.createCorrelationMatrices("mouse_matrix.h5", outFolder, clusterCount=clusterNumber, sampleCount=5000, verbose=True)
```

### II) Calculate average correlation of genes to gene sets for given gene set library

```python
import prismx as px

# reuse correlation matrices from first step
correlationFolder = "correlation_folder"
predictionFolder = "prediction_folder"
libs = px.listLibraries()

# select GO: Biological Processes
gmtFile = px.loadLibrary(libs[110])

px.correlationScores(gmtFile, correlationFolder, predictionFolder, verbose=True)
```

### III) Train model on GO: Biological Processes gene set library

```python
import prismx as px
import pickle

# reuse matrices from step I and step II
correlationFolder = "correlation_folder"
predictionFolder = "prediction_folder"
libs = px.listLibraries()

# select GO: Biological Processes (same as last step)
gmtFile = px.loadLibrary(libs[110])

model = px.trainModel(predictionFolder, correlationFolder, gmtFile, trainingSize=300000, testTrainSplit=0.1, samplePositive=40000, sampleNegative=200000, randomState=42, verbose=True)
pickle.dump(model, open("gobp_model.pkl", 'wb'))
```

Once the model is trained it can be applied on any gene set library of choice. Models trained with GO: BP were tested on all gene set libraries in Enrichr and show on average for all gene set libraries.