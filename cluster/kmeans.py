import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
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
        num_iters = 0
        centroids = mat[np.random.choice(mat.shape[0], size=self.k, replace=False)]
        k = self.k
        if self.centroid_distances == 0:
            self.centroid_distances = np.zeros([mat.shape[0], k])
        #centroids = np.random.rand(self.k, mat.shape[1])
        cluster_membership = np.zeros(mat.shape[0])
        while (self.max_iter > num_iters):
            for obs in range(distances.shape[0]):
                centroid_distances = np.zeros(k)
                for k in range(centroids.shape[0]):
                    centroid_distances[k] = np.sqrt(np.sum(np.square(mat[obs] - centroids[k])))
                cluster_membership[obs] = np.argmin(centroid_distances)
                self.centroid_distances[obs] = centroid_distances
            # Now, re-calculate centroid locations
            for k in range(centroids.shape[0]):
                centroids[k] = np.mean(mat[[i for i in cluster_membership if i == k]], axis = 0)
            num_iters += 1
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
        cluster_membership = np.zeros(mat.shape[0])
        for obs in range(distances.shape[0]):
            centroid_distances = np.zeros(k)
            for k in range(centroids.shape[0]):
                centroid_distances[k] = np.sqrt(np.sum(np.square(mat[obs] - centroids[k])))
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
