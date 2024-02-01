# Write your k-means unit tests here
import pytest
import cluster
from cluster import Silhouette
from cluster import KMeans
from cluster.utils import make_clusters
from sklearn.metrics.cluster import adjusted_rand_score

def test_kmeans():
    X, labels = make_clusters(n=1000, m=20, k=3)
    km = KMeans(k=3, tol = 1e-6, max_iter = 400)
    km.fit(X)
    cluster_membership = km.predict(X)
    ari = adjusted_rand_score(labels, cluster_membership)
    print("ARI: ", ari)
    assert ari > 0.8
