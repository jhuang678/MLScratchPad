import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from scipy.sparse import csgraph
#####################################################################

class ISOMAPLearner(object):

    def __init__(self, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print("\nInitialized ISOMAP Learner")

    def author(self):
        return "jhuang678"

    def compute_adjacency_matrix(self, X, epsilon=10, A_image=False):
        m = X.shape[0]
        self.A = np.zeros((m, m))

        for i in range(m):
            for j in range(i + 1, m):
                distance = np.linalg.norm(X[i] - X[j])  # Euclidean distance between data points
                if distance <= epsilon:
                    self.A[i, j] = distance
                    self.A[j, i] = distance

        if A_image:
            A_fig = plt.figure()
            ax = A_fig.gca()
            plt.spy(self.A)
            plt.title('Adjacency Matrix')
            for i in np.arange(0, 691, 70):
                file = 'output/faces/' + str(i) + '.png'
                face = image.imread(file)
                imagebox = OffsetImage(face, zoom=0.035)
                ab = AnnotationBbox(imagebox, (0+i, 698), frameon=False)
                ac = AnnotationBbox(imagebox, (698, 0+i), frameon=False)
                ax.add_artist(ab)
                ax.add_artist(ac)
                for j in np.arange(0, 691, 70):
                    if self.A[i, j] > 0:
                        ax.plot(i, j, 'ro')
                        ax.plot(j, i, 'ro')


            # Set axis names and title
            ax.set_xlabel('Nodes')
            ax.set_ylabel('Nodes')
            plt.savefig('output/adjacency_matrix.png')

    def compute_shortest_distance_matrix(self):
        self.D = csgraph.shortest_path(self.A)
        print(self.D)

    def compute_centering_matrix(self, X):
        m = X.shape[0]
        ones = np.ones((m, 1))
        I = np.eye(m)
        H = I - (1 / m) * ones @ ones.T
        self.C = (-1 / 2) * H @ self.D ** 2 @ H

    def compute_eigenvectors(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.C)
        return eigenvalues.real, eigenvectors.real

    def select_top_k_eigenvectors(self, eigenvalues, eigenvectors, k):
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top_k_indices = sorted_indices[:k]
        return eigenvalues[top_k_indices], eigenvectors[:, top_k_indices]

    def compute_reduced_represent(self, top_eigenvalues, top_eigenvectors):
        eigenvalue_matrix = np.diag(np.sqrt(top_eigenvalues))
        self.Z = top_eigenvectors @ eigenvalue_matrix


    def train_data(self, X, k=2, epsilon=10, A_image=False):
        m, d = X.shape
        print("Number of data:", m, "With dimensions =", d)
        print("K =", k)
        self.compute_adjacency_matrix(X=X, epsilon=epsilon, A_image=A_image)
        self.compute_shortest_distance_matrix()
        self.compute_centering_matrix(X=X)
        LAMBDA, W = self.compute_eigenvectors()
        lambdas, ws = self.select_top_k_eigenvectors(eigenvalues=LAMBDA, eigenvectors=W, k=k)
        self.compute_reduced_represent(top_eigenvalues=lambdas, top_eigenvectors=ws)
        return self.Z




if __name__ == "__main__":
    print("This is a ISOMAP Dimension Reduction Machine Learner.")


