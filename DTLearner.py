import numpy as np

class DTLearner(object):
    """
    :param verbose: If “verbose” is True, it can print out information for debugging.
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size:int = 1, verbose:bool = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        if self.verbose:
            print("\nInitialized Decision Tree:")
            print("Author: ",self.author())
            print("Leaf Size Setting: ",self.leaf_size)
            print("Please use 'fit(data_x, data_y)' to train a model.",'\n')

    def author(self):
        """
        :rtype: str  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        return "jhuang678"

    def best_feature(self, data_x:float, data_y:float):

        if data_x.size == 0 or data_y.size == 0:
            raise ValueError("Data cannot be empty.")

        col_index_list = []

        for i in range(data_x.shape[1]):
            if (np.std(data_x[:, i]) == 0):
                col_index_list.append(0)
            else:
                col_index_list.append(abs(np.corrcoef(data_x[:, i], data_y)[0, 1]))
        return col_index_list.index(max(col_index_list))

    def build_tree(self, data_x:float, data_y:float):
        if (data_x.shape[0] <= self.leaf_size):
            return np.array([np.nan, np.mean(data_y), np.nan, np.nan])

        elif (np.all(data_y == data_y[0])):
            return np.array([np.nan, data_y[0], np.nan, np.nan])

        else:
            best_x = self.best_feature(data_x, data_y)

            split_value = np.median(data_x[:,best_x])

            left_data = data_x[:, best_x] <= split_value
            right_data = data_x[:, best_x] > split_value

            if np.all(~right_data):
                return np.array([np.nan, np.mean(data_y), np.nan, np.nan])

            left_tree = self.build_tree(data_x[left_data], data_y[left_data])
            right_tree = self.build_tree(data_x[right_data], data_y[right_data])

            if left_tree.ndim == 1:
                root = np.array([best_x, split_value, 1, 2])
            else:
                root = np.array([best_x, split_value, 1, left_tree.shape[0] + 1])
            return np.row_stack((root, left_tree, right_tree))

    def fit(self, data_x:float, data_y:float):
        """
        Add training data to learner
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        self.tree = self.build_tree(data_x, data_y)
        if self.verbose:
            #np.set_printoptions(threshold=sys.maxsize)
            print('Trained Decision Tree:')
            print('Tree Size:', self.tree.shape)
            print("Please use 'query(points)' to predict values.")

    def query(self, points:float):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		  	  		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  	  		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        tree = self.tree

        if points.ndim == 1:
            point = points
            node = 0
            node_info = tree[node]
            while ~np.isnan(node_info[0]):
                if point[int(node_info[0])] <= node_info[1]:
                    node = node + int(node_info[2])
                else:
                    node = node + int(node_info[3])
                node_info = tree[node]
            return np.array(node_info[1])

        else:
            pred_y = np.array([])
            for point in points:
                node = 0
                node_info = tree[node]
                while ~np.isnan(node_info[0]):
                    if (point[int(node_info[0])] <= node_info[1]):
                        node = node + int(node_info[2])
                    else:
                        node = node + int(node_info[3])
                    node_info = tree[node]
                pred_y = np.append(pred_y, np.array(node_info[1]))
            return pred_y

if __name__ == "__main__":
    print("This is DTLearner.py. Please use 'DTLearner(leaf_size, verboseCan you shoe )' to initialize.")
