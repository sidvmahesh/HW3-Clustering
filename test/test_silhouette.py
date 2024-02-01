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
    scores = Silhouette().score(X, labels)
    print("ground_truth: ", ground_truth)
    print("scores: ", scores)
    assert(len(ground_truth) == len(scores))
    for i in range(len(ground_truth)):
        if abs(ground_truth[i] - scores[i]) > 0.02:
            print("ERROR: observation", i, "has incorrect silhouette score")
            assert False
    assert True

    

