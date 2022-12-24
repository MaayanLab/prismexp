![prismxsmall](https://user-images.githubusercontent.com/32603869/209446641-47383bdd-95a5-4dd9-b6ca-50d5f3b82936.png#gh-dark-mode-only | width=160)
![prismxsmall_light](https://user-images.githubusercontent.com/32603869/209446724-d5bf8be7-6d74-466b-9684-fc49007383d5.png#gh-light-mode-only | width=160)

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

## Quick Usage Example

```python
import urllib.request
import prismx as px

urllib.request.urlretrieve("https://s3.dev.maayanlab.cloud/archs4/archs4_gene_human_v2.1.2.h5", "human_matrix.h5")

work_dir = "/home/maayanlab/code/prismexp"
h5_file = "human_matrix.h5"
gmt_file = px.load_library("GO_Biological_Process_2021")

cluster_number = 100

px.create_correlation_matrices(h5_file, work_dir, cluster_count=cluster_number, verbose=True)
px.features(gmt_file, work_dir, threads=4, verbose=True)
px.train(work_dir, gmt_file, verbose=True)
px.predict(work_dir, gmt_file, verbose=True)
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

This is the first step of PrismExp. We first identify N = cluster_numer clusters of samples for the ARCHS4 gene expression. We then limit each cluster to maximally sample_count = 5000 samples. After normalizing the gene expression PrismExp computes the pairwise correlation of genes in each cluster resulting in N correlation matrices.

```python
import urllib.request
import prismx as px

urllib.request.urlretrieve("https://s3.dev.maayanlab.cloud/archs4/archs4_gene_human_v2.1.2.h5", "human_matrix.h5")

work_dir = "/home/maayanlab/code/prismexp/"
h5_file = "human_matrix.h5"

cluster_number = 100

px.create_correlation_matrices(h5_file,
                               work_dir,
                               cluster_count=cluster_number, 
                               sample_count=5000, 
                               cluster_gene_count=1000,
                               reuse_clustering=False,
                               verbose=True)
                               
```

### create_correlation_matrices

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| h5_file | str | | The path to the h5 file containing the gene expression data. |
| work_dir | str | | The directory to save the resulting clustering and correlation matrices. |
| cluster_count | int | 100 | The number of clusters to use for the sample clustering. |
| read_threshold | int | 20 | The minimum number of reads a gene must have in a fraction of total reads to keep. |
| sample_threshold | float | 0.01 | The minimum fraction of samples that contain `read_threshold` reads of a gene to keep. |
| filter_samples | int | 2000 | The maximum number of samples to use for gene filtering. |
| min_avg_reads_per_gene | int | 2 | The average number of reads per gene for a sample to be considered in the clustering. Can be used to remove samples with very low library size. |
| cluster_method | str | "minibatch" | The clustering method to use. Options are "minibatch" and "kmeans". minibatch is much faster.|
| cluster_gene_count | int | 1000 | The number of genes to use for the sample clustering. |
| sample_count | int | 5000 | The maximum number of samples to use for calculating the correlation matrices. |
| reuse_clustering | bool | False | Whether to reuse the existing clustering results in the work directory. |
| correlation_method | str | "pearson" | The correlation method to use. Options are "pearson" and "spearman". Pearsons correlation is faster and requires less memory. |
| verbose | bool | True | Whether to print progress messages. |


### II) Calculate average correlation of genes to gene sets for given gene set library

This is the feature generation step. PrismExp will iterate over the previously generated correlation matrices and compute the average correlation (features) for the given gene set library. Features are required for model training and also prediction. For training a good library is GO Biological Processes.

```python
import prismx as px

work_dir = "/home/maayanlab/code/prismexp/"

# load Enrichr library to use
gmt_file = px.load_library("GO_Biological_Process_2021")

# calculate the features that are used for model training and prediction
px.features(gmt_file, work_dir, threads=4, verbose=True)
```

#### features

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| gmt_file | str | | Path to the gmt file containing the gene set library. |
| work_dir | str | | Path to the directory containing the correlation matrices. |
| intersect | bool | False | If True, only includes unique genes present in all gene sets in the feature matrix. |
| threads | int | 2 | Number of threads to use for parallel processing. |
| verbose | bool | False | If True, prints progress information. |

### III) Train model on GO: Biological Processes gene set library

The gene set library needs to be the same as the one used in the prior feature generation step.

```python
import prismx as px

work_dir = "/home/maayanlab/code/prismexp/"

gmt_file = px.load_library("GO_Biological_Process_2021")

# build a training data set and train model
model = px.train(work_dir, gmt_file, training_size=300000, 
            test_train_split=0.1, sample_positive=40000,
            sample_negative=200000, random_state=1, verbose=True)
```

### train

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| work_dir | str | | Path to the directory containing the correlation matrices. |
| gmt_file | str | | Path to the gmt file containing the gene set library. |
| training_size | int | 200000 | The number of gene sets to use for training. |
| test_train_split | float | 0.1 | The proportion of the training data to use for testing. |
| sample_positive | int | 20000 | The number of positive samples to use in the balanced training data. |
| sample_negative | int | 80000 | The number of negative samples to use in the balanced training data. |
| random_state | int | 42 | The seed for the random number generator. |
| verbose | bool | False | If True, prints progress information. |

Once the model is trained it can be applied on any gene set library of choice. Models trained with GO: BP were tested on all gene set libraries in Enrichr and show on average for all gene set libraries.

### IV) Predict gene functions

For the prediction step the model can be used across different libraries. There is also very low risk of overfitting the model so it can be trained and applied on the same gene set library. In this example the model was trained in BO Biological Processes, but applied on KEGG pathways. The prediction step will recompute the features, unless explicitly instructed to reuse the features. The prediction is saved as feather file at `{work_dir}/predictions/{gmt_file}.f`

```python
import prismx as px

work_dir = "/home/maayanlab/code/prismexp/"

gmt_file = px.load_library("KEGG_2021_Human")

px.predict(work_dir, gmt_file, step_size=500, verbose=True)
```

To read the prediction matrix (genes as rows and gene sets as columns):

```python3
import pandas as pd
import feather

work_dir = "/home/maayanlab/code/prismexp/"

predictions = pd.read_feather(work_dir+"/predictions/KEGG_2021_Human.f").set_index("index")
```

#### predict

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| work_dir | str | | Path to the directory containing the correlation matrices and precomputed model. |
| gmt_file | str | | Path to the gmt file containing the gene set library. |
| model | lightGBM model | `None` | The prediction model to use. If `None`, loads the model from the workdir. |
| step_size | int | 1000 | The number of samples to process at a time. |
| intersect | bool | False | If True, only includes unique genes present in all gene sets in the feature matrix. |
| normalize | bool | False | If True, normalizes the final prediction values using a z-score. |
| verbose | bool | False | If True, prints progress information. |
| skip_features | bool | False | If True, skips the feature computation step. |
| threads | int | 2 | Number of threads to use for parallel processing. |

## Bridge Gene Set Enrichment Analysis (bridgeGSEA)

Use PrismEXP gene set predictions in enrichment analysis to identify novel genes in enriched pathways and biological processes.

```python
import prismx as px
import prismx.gsea as pxgsea
import blitzgsea as blitz
import urllib.request
import feather
import pandas as pd

work_dir = "/home/maayanlab/code/prismexp/"

url = "https://github.com/MaayanLab/blitzgsea/raw/main/testing/ageing_muscle_gtex.tsv"
urllib.request.urlretrieve(url, "ageing_muscle_gtex.tsv")

# read signature as pandas dataframe
signature = pd.read_csv("ageing_muscle_gtex.tsv")

# use enrichr submodule to retrieve gene set library
library = blitz.enrichr.get_library("GO_Biological_Process_2021")

# load PrismExp predictions
predictions = pd.read_feather(work_dir+"/predictions/GO_Biological_Process_2021.f").set_index("index")

result = px.bridgegsea.bridge_gsea(signature, library, predictions)
```
