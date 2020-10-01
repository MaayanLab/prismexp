import h5py as h5
import pandas as pd
import numpy as np
import os

f = h5.File("human_tpm_v8.h5", 'r')
expression = f['data/expression']
fields = f["meta"]
samples = f['meta/Sample_geo_accession']
genes = f['meta/genes']


