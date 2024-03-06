import numpy as np

class PCALearner(object):

    def __init__(self, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print("\nInitialized PCA Learner")

    def author(self):
        return "jhuang678"

    def standardize(self, X):
        # Subtract the mean and divide by standard deviation
        X_std = np.std(X, axis=0)
        if np.any(X_std == 0):
            return X - np.mean(X, axis=0)
        else:
            return (X - np.mean(X, axis=0)) / X_std

    def compute_covariance_matrix(self, X):
        m = X.shape[0]
        return (1 / m) * np.dot(X.T, X)

    def compute_eigenvectors(self, cov_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        return eigenvalues.real, eigenvectors.real

    def select_top_k_eigenvectors(self, eigenvalues, eigenvectors, k):
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top_k_indices = sorted_indices[:k]
        return eigenvalues[top_k_indices], eigenvectors[:, top_k_indices]

    def compute_reduced_represent(self, X, top_eigenvalues, top_eigenvectors):
        Z = np.dot(top_eigenvectors.T, X.T) / np.sqrt(top_eigenvalues[:, np.newaxis])
        return Z.T

    def train_data(self, X, k=2):
        m, d = X.shape
        print("Number of data:", m, "With dimensions =", d)
        print("K =", k)

        # Step 1: Standardize the data if required
        X = self.standardize(X)

        # Step 2: Compute the covariance matrix
        C = self.compute_covariance_matrix(X)

        # Step 3: Compute the eigenvalues and eigenvectors
        LAMBDA, W = self.compute_eigenvectors(cov_matrix=C)

        # Step 4: Select the top-k eigenvectors
        lambdas, ws = self.select_top_k_eigenvectors(eigenvalues=LAMBDA, eigenvectors=W, k=k)

        # Step 5: Compute reduced representation
        Z = self.compute_reduced_represent(X=X, top_eigenvalues=lambdas, top_eigenvectors=ws)

        return Z




if __name__ == "__main__":
    print("This is a PCA Dimension Reduction Machine Learner.")

