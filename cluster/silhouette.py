import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        silhouette_scores = np.zeros(X.shape[0])
        for obs in range(silhouette_scores.shape[0]):
            points_of_same_cluster = X[[i for i in range(y.shape[0]) if (y[i] == y[obs] and i != obs)]]
            a = np.mean(cdist(np.reshape(X[obs], (1, X.shape[1])), points_of_same_cluster))
            #a = np.mean(np.sqrt(np.sum(np.square(points_of_same_cluster - X[obs])), axis = 1))
            cluster_distances = []
            unique_clusters = [i for i in np.unique(y) if i != y[obs]]
            for other_cluster in unique_clusters:
                points_of_other_cluster = X[[i for i in range(y.shape[0]) if y[i] == other_cluster]]
                #cluster_distances.append(np.mean(np.sqrt(np.sum(np.square(points_of_other_cluster - X[obs])), axis = 1)))
                cluster_distances.append(cdist(np.reshape(X[obs], (1, X.shape[1])), points_of_other_cluster))
            b = np.min(cluster_distances)
            silhouette_scores[obs] = (b - a) / (max(b, a))
        return silhouette_scores
