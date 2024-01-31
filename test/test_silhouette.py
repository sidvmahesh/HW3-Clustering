# write your silhouette score unit tests here
import pytest
import cluster
from cluster import Silhouette
from cluster import KMeans
from cluster.utils import make_clusters
from sklearn.metrics import silhouette_samples

def test_silhouette_score():
    X, labels = make_clusters(n=1000, m=20, k=3)
    ground_truth = silhouette_samples(X, labels)
    scores = Silhouette.score(X, labels)
    assert ground_truth == scores

    

