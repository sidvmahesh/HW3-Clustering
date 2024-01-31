# Write your k-means unit tests here
import pytest
import cluster
from cluster import Silhouette
from cluster import KMeans
from sklearn.metrics import silhouette_samples

def test_kmeans():
    X, labels = make_clusters(n=1000, m=20, k=3)
    km = KMeans(k=3)
    km.fit(X)
    cluster_membership = km.predict(X)
    scores = Silhouette.score(X, labels)
    assert ground_truth == scores
    