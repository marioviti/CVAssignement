# Cluster candidate samples and remove those that do not belong to clusters with at least 3 samples
# generate 3D feature points (center_i, center_j, width)
from sklearn.cluster import DBSCAN
import numpy as np

def get_clusters(candidates):
  X = np.array( [ [row, (strat+end)/2, end-strat]  for row,strat,end in candidates ] )
  Xw = (X - X.mean())/ X.std()
  #X = X / X.std()
  clustering = DBSCAN(eps=0.1, min_samples=3).fit(Xw)
  clusters = [ X[clustering.labels_==c].mean(axis=0) for c in range(np.max(clustering.labels_)+1) ]
  return clusters
