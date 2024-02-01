import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 1000):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if (type(k) != int) or (k < 1):
            print("k must be a positive integer")
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = 0
        self.centroid_distances = 0

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        centroids = mat[np.random.choice(mat.shape[0], size=self.k, replace=False)]
        print("centroids: ", centroids.shape)
        k = self.k
        if self.centroid_distances == 0:
            self.centroid_distances = np.zeros([mat.shape[0], k])
        cluster_membership = np.zeros(mat.shape[0])
        print("cluster_membership: ", cluster_membership.shape)
        print("self.centroid_distances: ", self.centroid_distances.shape)
        for num_iter in range(self.max_iter):
            for obs in range(self.centroid_distances.shape[0]):
                centroid_distances = cdist(np.reshape(mat[obs], (1, mat.shape[1])), centroids)
                cluster_membership[obs] = np.argmin(centroid_distances)
                self.centroid_distances[obs] = centroid_distances
            # Now, re-calculate centroid locations
            for k_centroid in range(centroids.shape[0]):
                centroids[k_centroid] = np.mean(mat[np.argwhere(cluster_membership == k_centroid)], axis = 0)
            # Finally, check if the clustering has converged
            cumulative_error = 0
            for i in range(centroids.shape[0]):
                cumulative_error += cdist(np.reshape(centroids[i], (1, centroids.shape[1])), np.reshape(self.centroids[i], (1, self.centroids.shape[1])))
            if cumulative_error < self.tol:
                break # Early convergence
            self.centroids = centroids

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        print("self.centroids: ", self.centroids.shape)
        cluster_membership = np.zeros(mat.shape[0])
        for obs in range(self.centroid_distances.shape[0]):
            centroid_distances = np.zeros(self.k)
            centroid_distances = cdist(np.reshape(mat[obs], (1, mat.shape[1])), self.centroids)
            cluster_membership[obs] = np.argmin(centroid_distances)
        return cluster_membership


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        squared_diffs = np.square(self.centroid_distances)
        closest_centroids = np.argmin(squared_diffs, axis=1)
        return np.sum(closest_centroids)


    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids
