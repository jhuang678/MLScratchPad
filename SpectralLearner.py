import numpy as np
import KMeansLearner as km

class SpectralLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print("\nInitialized Spectral Cluster")

    def author(self):
        return "jhuang678"

    def create_adjacency_matrix(self, nodes, edges):
        num_nodes = len(nodes)
        self.A = np.zeros((num_nodes,num_nodes))
        for edge in edges:
            node1, node2 = edge
            self.A[node1 - 1, node2 - 1] = 1
            self.A[node2 - 1, node1 - 1] = 1
        return self.A

    def create_degree_matrix(self, nodes=[], edges=[]):
        if hasattr(self, 'A'):
            D_vector = np.sum(self.A, axis=1)  # Calculate row sums
            self.D = np.diag(D_vector)  # Create a diagonal matrix with row sums as diagonal elements
        else:
            max_node = np.max(nodes)
            self.D = np.zeros((max_node, max_node))
            for edge in edges:
                node1, node2 = edge
                self.D[node1 - 1, node1 - 1] += 1
                self.D[node2 - 1, node2 - 1] += 1
        return self.D

    def create_Laplacian_matrix(self):
        self.L = self.D - self.A
        return self.L

    def compute_eigenvectors(self, k):
        eigenvalues, eigenvectors = np.linalg.eig(self.L)
        sorted_indices = np.argsort(eigenvalues)
        self.Z = np.real(eigenvectors[:, sorted_indices[:k]])
        return self.Z

    def kmeans(self, k, norm=2):
        learner = km.KmeansLearner(verbose=self.verbose)
        labels, CENTROID = learner.train_data(self.Z, k=k, norm=norm, max_iterations=1000)
        return labels

    def train_data(self, nodes, edges, k, norm=2, dircected=False):
        nodes_map = {node: index for index, node in enumerate(nodes)}
        new_nodes = np.array([nodes_map[node] for node in nodes])
        new_edges = np.array([[nodes_map[node1], nodes_map[node2]] for node1, node2 in edges])

        self.create_adjacency_matrix(nodes=new_nodes , edges=new_edges)
        self.create_degree_matrix()
        self.create_Laplacian_matrix()
        self.compute_eigenvectors(k=k)
        labels = self.kmeans(k=k, norm=norm)
        print("Number of nodes:", new_nodes.shape[0], ";Number of edges: = ", new_edges.shape[0])
        return labels

if __name__ == "__main__":
    print("This is a Spectral Cluster Machine Learner.")
    print("start testing!")

    learner = SpectralLearner(verbose=True)
    nodes = np.array([1,2,3,4])
    edges = np.array([[1,2],[2,3],[3,4]])
    predict_labels = learner.train_data(nodes=nodes, edges=edges, k=2)




