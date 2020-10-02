from sklearn.cluster import KMeans
import h5py as h5
import numpy as np
import pandas as pd
import random
from typing import List
import sys
import os
import time

os.remove("mouse_matrix_2.h5")

f1 = h5.File("mouse_matrix.h5", "r")
exp = f1["data/expression"]


f = h5.File("mouse_matrix_2.h5", "w")
dset = f.create_dataset("data/expression", exp.shape, chunks=(1, 2000), dtype=np.int32, compression='gzip', compression_opts=9)


steps = 500
stepSize = math.floor(exp.shape[0]/steps)

steps = 5
for i in range(0, steps):
    print(i)
    fromStep = i*stepSize
    toStep = (i+1)*stepSize
    ee = exp[fromStep:toStep, :]
    dset[fromStep:toStep, :] = exp[fromStep:toStep, :]

f.close()
f1.close()

f = h5.File("mouse_matrix_2.h5", "r")

sa = random.sample(set(range(0, 5000)), 1000).sorted()

start = time.time()
exp = f1["data/expression"][sa, :]
print(time.time()- start)




f1 = h5.File("mouse_matrix.h5", "r")
f = h5.File("mouse_matrix_3.h5", "w")

keys = list(f1["meta"].keys())

for k in keys:
    print(k)
    f.create_dataset("meta/"+k, data=f1["meta/"+k], )

f.close()
f1.close()


f1 = h5.File("mouse_matrix.h5", "r")
exp = f1["data/expression"]

f = h5.File("mouse_matrix_2.h5", "w")


f.close()
f1.close()

