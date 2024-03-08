import numpy as np

class KNNLearner(object):
    def __init__(self, k=3, verbose=False):
        self.k = k
        self.verbose = verbose
        if self.verbose:
            print("\nInitialized K-Nearest Neighbors Learner:")
            print("Author: ", self.author())
            print("Please use 'fit(data_x, data_y)' to train the model.")

    def author(self):
        return "jhuang678"

    def fit(self, data_x, data_y):
        # Store the training data
        self.data_x = data_x
        self.data_y = data_y
        self.is_classification = len(set(data_y)) == 2

        if self.verbose:
            print("\nTrained K-Nearest Neighbors:")
            print("Number of Neighbors: ", self.k)
            print("Classification Mode: ", self.is_classification)
            print("Please use 'query(points)' to make predictions.")

    def query(self, points):
        predictions = []

        for point in points:
            distances = np.sqrt(((self.data_x - point) ** 2).sum(axis=1))

            neighbors_indices = distances.argsort()[:self.k]

            if self.is_classification:
                neighbor_labels = self.data_y[neighbors_indices]
                prediction = max(set(neighbor_labels), key=list(neighbor_labels).count)
            else:
                neighbor_values = self.data_y[neighbors_indices]
                prediction = np.mean(neighbor_values)

            predictions.append(prediction)

        return np.array(predictions)

if __name__ == "__main__":
    print("This is KNNLearner.py. Please use 'KNNLearner(verbose=True)' to initialize.")
