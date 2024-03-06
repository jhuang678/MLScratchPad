import numpy as np
import time
class KmeansLearner(object):

    def __init__(self, verbose = False):
        self.verbose = verbose
        if self.verbose:
            print("\nInitialized K-means Cluster")

    def author(self):
        return "jhuang678"

    def random_initialize_centroids(self, X, k):
        int_centers = np.random.choice(len(X), size=k, replace=False)
        return X[int_centers, :]

    def compute_distance(self, X, centers, norm):
        distance_matrix = np.linalg.norm(X[:, np.newaxis, :] - centers, ord=norm, axis=2)
        return distance_matrix

    def assign_data_to_clusters(self, distance_matrix):
        cluster_labels = np.argmin(distance_matrix, axis=1)
        return cluster_labels

    def assign_centers(self, X, k, cluster_labels, norm):
        centers = np.empty((k, X.shape[1]))
        if (norm == 1):
            for j in range(k):
                if np.sum(cluster_labels == j) > 0:  # Check if cluster is not empty
                    centers[j, :] = np.median(X[cluster_labels == j, :], axis=0)
                else:
                    centers[j, :] = np.mean(X, axis=0)  # Use the overall mean if cluster is empty
        else:
            for j in range(k):
                if np.sum(cluster_labels == j) > 0:  # Check if cluster is not empty
                    centers[j, :] = np.mean(X[cluster_labels == j, :], axis=0)
                else:
                    centers[j, :] = np.mean(X, axis=0)  # Use the overall mean if cluster is empty
        return centers

    def has_converged(self, old_centers, centers):
        return set([tuple(x) for x in old_centers]) == set([tuple(x) for x in centers])

    def calculate_total_distance(self, X, centers, cluster_labels):
        distance_sum = 0
        for i in range(len(X)):
            data_point = X[i]
            cluster_label = cluster_labels[i]
            center = centers[cluster_label]
            distance_sum += np.linalg.norm(data_point - center, ord=2)
        return distance_sum

    def train_data(self, X, k, norm=2, max_iterations=150):
        # Set up parameters
        converged = False
        i = 0
        # (1) Random Initialize Centroids
        centers = self.random_initialize_centroids(X, k)
        ts = time.time()
        while (not converged) and (i <= max_iterations):
            old_centers = centers

            # (2) Computing the Distances
            distance_matrix = self.compute_distance(X, centers, norm)

            # (3) Assign Data Points to Clusters
            cluster_labels = self.assign_data_to_clusters(distance_matrix)

            # (4) Assign New Centers
            centers = self.assign_centers(X, k, cluster_labels, norm)

            # (5) Assign New Centers
            converged = self.has_converged(old_centers, centers)
            i += 1
        ts = time.time() - ts
        m, d = X.shape
        print("Number of data:", m, "With dimensions = ", d)
        print("K = ", k, "; Norm = ", norm)
        print("iteration", i, "; Cumulative Time = ", ts)
        if (i > max_iterations):
            print("Iteration exceed max iteration.")
        total_distance = self.calculate_total_distance(X, centers, cluster_labels)
        print("Total distance:", total_distance)
        print()
        return cluster_labels, centers

if __name__ == "__main__":
    print("This is a K-Means Cluster Machine Learner.")

