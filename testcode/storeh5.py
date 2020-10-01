import h5py as h5
import feather
import pandas as pd
import numpy as np
import os

correlation_files = os.listdir("correlation_folder")

for i in range(0, len(correlation_files)):
    print(i)
    correlation = pd.read_feather("correlation_folder/"+correlation_files[i])
    f = h5.File("h5/"+correlation_files[i].replace(".f","")+".h5", "w")
    dset = f.create_dataset("data/correlation", correlation.shape, dtype=np.float16, chunks=(1,correlation.shape[0]))
    dset[:,:] = correlation
    genemeta = f.create_dataset("meta/genes", data=np.array(list(map(str.upper, correlation.columns)), dtype='S10'), dtype='S10')
    f.close()


def loadCorrelation(gene):
    f = h5.File("h5/correlation_2.h5", "r")
    genes = np.array(f["meta/genes"]).astype(np.str)
    idx = list(genes).index(gene)
    cor = np.array(f["data/correlation"][:,idx]).astype(np.float64)
    f.close()
    return(cor)

start = time.time()
coco = loadCorrelation("SOX2")
print(time.time() - start)




import h5py as h5
import s3fs
import numpy as np
import time
from multiprocessing import Process
import random

def loadGenesS3():
    genes = 0
    s3 = s3fs.S3FileSystem(anon=True)
    with h5.File(s3.open("s3://mssm-prismx/correlation_0.h5", 'rb'), 'r', lib_version='latest') as f:
        genes = np.array(f["meta/genes"]).astype(np.str)
    return genes

def loadCorrelationS3(gene, genes, cormat, results):
    cor = 0
    s3 = s3fs.S3FileSystem(anon=True)
    with h5.File(s3.open("s3://mssm-prismx/correlation_"+str(cormat)+".h5", 'rb'), 'r', lib_version='latest') as f:
        idx = list(genes).index(gene)
        cor = np.array(f["data/correlation"][idx,:]).astype(np.float64)
        results[cormat] = cor

genes = loadGenesS3()

start = time.time()
coco = loadCorrelationS3("MAPK1", genes)
print(time.time() - start)


from multiprocessing.pool import ThreadPool as Pool
import pandas as pd



start = time.time()
pool = Pool(1)
cormats = list(range(0,50))
cormats.append("global")
results = pd.DataFrame(np.zeros(shape=(len(genes), len(cormats))), columns=cormats)
for i in cormats:
    pool.apply_async(loadCorrelationS3, ("P53", genes, i, results))

pool.close()
pool.join()
print(time.time() - start)



start = time.time()
pool = Pool(10)
results = pd.DataFrame(np.zeros(shape=(len(genes), 20)), columns=genes[1000:1020])
for gene in genes[1000:1010]:
    results[gene] = pool.apply_async(loadCorrelationS3, (gene, genes,)).get()

pool.close()
pool.join()
print(time.time() - start)

start = time.time()
for gene in genes[2000:2050]:
    loadCorrelationS3(gene, genes, results)

print(time.time() - start)


f = h5.File("h5/correlation_0.h5", "r")
genes = np.array(f["meta/genes"]).astype(np.str)
f.close()

idx = list(genes).index("0610009L18")
print(idx)

list(genes).index('0610009L18')


f = h5.File("h5/correlation_0.h5", "r")

genes = np.array(f["meta/genes"]).astype(np.str)

f.close()


f = h5.File("h5/correlation_0.h5", "w")
dset = f.create_dataset("data/correlation", correlation.shape, dtype=np.float16, chunks=(1,correlation.shape[0]))
dset[:,:] = correlation
genemeta = f.create_dataset("meta/genes", data=np.array(list(map(str.upper, correlation.columns)), dtype='S10'), dtype='S10')
f.close()


import numpy as np
import matplotlib.pyplot as plt
import numpy as np

t1 = np.arange(0, 8, 0.001)

s1 = np.sin(t1) + 1.5
s2 = np.sin(t1*6)/5
s3 = s1+s2-3.5

g1 = np.sin(t1+np.pi/2) + 1.5
g2 = np.sin(t1*6)/5
g3 = g1+g2-3.5


plt.plot(t1, s1, label="low frequency")
plt.plot(t1, s2, label="high frequency")
plt.plot(t1, s3, label="combined frequency")
plt.legend()
plt.title("gene A")
#plt.show()
plt.savefig("genea.png")
plt.close()


plt.plot(t1, g1, label="low frequency")
plt.plot(t1, g2, label="high frequency")
plt.plot(t1, g3, label="combined frequency")
plt.legend()
plt.title("gene B")
#plt.show()
plt.savefig("geneb.png")
plt.close()

plt.plot(t1, s3+3.5, label="gene A")
plt.plot(t1, g3+3.5, label="gene B")
plt.legend()
plt.title("full spectrum gene similarity")
#plt.show()
plt.savefig("fullspectrum.png")
plt.close()

plt.plot(t1, s2, label="gene A")
plt.plot(t1, g2, label="gene B")
plt.legend()
plt.title("high frequency spectrum gene similarity")
#plt.show()
plt.savefig("highspectrum.png")
plt.close()

np.corrcoef(s3,g3)


k1 = list(s3[4000:8000])+list(s3[0:4000])
k2 = list(g3[4000:8000])+list(g3[0:4000])

plt.plot(t1, np.array(k1)+3.5, label="gene A")
plt.plot(t1, np.array(k2)+3.5, label="gene B")
plt.legend()
plt.title("shuffled spectrum gene similarity")
#plt.show()
plt.savefig("shufflespectrum.png")
plt.close()

