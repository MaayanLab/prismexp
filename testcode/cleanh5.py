from sklearn.cluster import KMeans
import h5py as h5
import numpy as np
import pandas as pd
import random
from typing import List
import sys
import os
import time
import math

os.remove("mouse_matrix_3.h5")

f1 = h5.File("mouse_matrix.h5", "r")
exp = f1["data/expression"]


f = h5.File("mouse_matrix_3.h5", "w")
dset = f.create_dataset("data/expression", exp.shape, chunks=(2, 3000), dtype=np.int32, compression='gzip',  compression_opts=9)

steps = 500
stepSize = math.floor(exp.shape[0]/steps)

for i in range(0, steps+1):
    print(i)
    fromStep = i*stepSize
    toStep = min((i+1)*stepSize, exp.shape[0])
    ee = exp[fromStep:toStep, :]
    dset[fromStep:toStep, :] = exp[fromStep:toStep, :]

f.close()
f1.close()



## benchmark me
f = h5.File("mouse_matrix_2.h5", "r")
sa = random.sample(set(range(0, 28000)), 2000)
sa.sort()
start = time.time()
exp = f["data/expression"][sa, :]
print("Extract samples: "+str(time.time()- start))
f.close()

f = h5.File("mouse_matrix_2.h5", "r")
sa = random.sample(set(range(0, 32000)), 10)
sa.sort()
start = time.time()
exp = f["data/expression"][:, 5]
print("Extract gene: "+str(time.time() - start))
f.close()

f = h5.File("mouse_matrix_2.h5", "r")
sa = random.sample(set(range(0, 32000)), 10)
sa.sort()
start = time.time()
exp = f["data/expression"][:, sa]
print("Extract gene (10): "+str(time.time() - start))
f.close()






f1 = h5.File("mouse_matrix.h5", "r")
f = h5.File("mouse_matrix_3.h5", "a")

keys = list(f1["meta"].keys())

for k in keys:
    print(k)
    f.create_dataset("meta/"+k, data=f1["meta/"+k], compression='gzip', compression_opts=9)

f.close()
f1.close()




f1 = h5.File("mouse_matrix.h5", "r")
exp = f1["data/expression"]

f = h5.File("mouse_matrix_2.h5", "w")


f.close()
f1.close()




## benchmark me
f = h5.File("mouse_matrix_t.h5", "r")
sa = random.sample(set(range(0, 284907)), 2000)
sa.sort()
start = time.time()
exp = f["data/expression"][sa, :]
print("Extract samples: "+str(time.time()- start))
f.close()

f = h5.File("mouse_matrix_t.h5", "r")
sa = random.sample(set(range(0, 32000)), 10)
sa.sort()
start = time.time()
exp = f["data/expression"][:, 5]
print("Extract gene: "+str(time.time() - start))
f.close()

f = h5.File("mouse_matrix_t.h5", "r")
sa = random.sample(set(range(0, 32000)), 10)
sa.sort()
start = time.time()
exp = f["data/expression"][:, sa]
print("Extract gene (10): "+str(time.time() - start))
f.close()




## benchmark me
f = h5.File("mouse_matrix.h5", "r")
sa = random.sample(set(range(0, 284907)), 2000)
sa.sort()
start = time.time()
exp = f["data/expression"][sa, :]
print("Extract samples: "+str(time.time()- start))
f.close()

f = h5.File("mouse_matrix.h5", "r")
sa = random.sample(set(range(0, 32000)), 10)
sa.sort()
start = time.time()
exp = f["data/expression"][:, 5]
print("Extract gene: "+str(time.time() - start))
f.close()

f = h5.File("mouse_matrix.h5", "r")
sa = random.sample(set(range(0, 32000)), 10)
sa.sort()
start = time.time()
exp = f["data/expression"][:, sa]
print("Extract gene (10): "+str(time.time() - start))
f.close()


## benchmark me
f = h5.File("mouse_matrix_t.h5", "r")
sa = random.sample(set(range(0, 284907)), 500)
sa.sort()
start = time.time()
exp = f["data/expression"][sa, :]
print("Extract samples: "+str(time.time()-start))
f.close()

f = h5.File("mouse_matrix.h5", "r")
sa.sort()
start = time.time()
exp2 = f["data/expression"][sa, :]
print("Extract samples: "+str(time.time()-start))
f.close()
