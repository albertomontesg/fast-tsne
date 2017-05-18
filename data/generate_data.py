import numpy as np
import struct
import sys

if len(sys.argv) < 4:
    print("Provide 3 arguments: number of clusters, number of dimensions, number of samples")
    exit()



N = int(sys.argv[3])
C = int(sys.argv[1]) #number of clusters
D = int(sys.argv[2]) #number of dimensions

fname = ""
if len(sys.argv) == 5: # where to store output
    fname += sys.argv[4]
fname += "%d_%d_%d" % (C, D, N)



#for each data point generate a cluster cluster
clusters = np.random.randint(0, C, size=(N))
#initialize the clusters
means = C*D*np.random.rand( C, D )
cov = np.random.rand( C, D, D )
for i in range(C):
    cov[i] = (cov[i].T @ cov[i])
    cov[i] = cov[i] / cov[i].max()
    cov[i] = ((1/(C*D))**2) * cov[i]


data = np.zeros( (N,D) )
for i in range(C):
    nrSamples = np.sum(clusters == i)
    samples = np.random.multivariate_normal(mean = means[i, :], cov=cov[i], size = nrSamples)
    data[np.where(clusters == i)[0], :] = samples

data = data - np.tile(data.min(axis=0).reshape(1,D), (N,1))
data = data / np.tile(data.max(axis=0).reshape(1,D), (N,1))
data = 255*data
data = data.astype(np.uint8)

with open(fname, 'wb') as w:
    w.write(struct.pack(">IIII", 2051, N, 1, D))
    w.write(memoryview(data))

clusters = clusters.astype(np.uint8)
with open(fname + '.labels', 'wb') as w:
    w.write(struct.pack(">II", 2049, N))
    w.write(memoryview(clusters))



