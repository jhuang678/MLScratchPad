import numpy as np

class LinRegLearner(object):
    """
    :param verbose: If “verbose” is True, it can print out information for debugging.
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False):
        self.verbose = verbose
        if self.verbose:
            print("\nInitialized Linear Regression:")
            print("Author: ", self.author())
            print("Please use 'fit(data_x, data_y)' to train a model.")


    def author(self):
        """
        :rtype: str  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        return "jhuang678"

    def fit(self, data_x:float, data_y:float):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		  	  		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		  	  		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		  	  		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        """

        # slap on 1s column so linear regression finds a constant term  		  	   		  	  		  		  		    	 		 		   		 		  
        new_data_x = np.ones([data_x.shape[0], data_x.shape[1] + 1])  		  	   		  	  		  		  		    	 		 		   		 		  
        new_data_x[:, 0 : data_x.shape[1]] = data_x

        # build and save the model
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(  		  	   		  	  		  		  		    	 		 		   		 		  
            new_data_x, data_y, rcond = None
        )

        if self.verbose:
            print("\nTrained Linear Regression:")
            print("Constant: ", self.model_coefs[-1])
            print("Coefficients: ", self.model_coefs[:-1])
            print("Please use 'query(points)' to predict values.")


  		  	   		  	  		  		  		    	 		 		   		 		  
    def query(self, points:float):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  	  		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[-1]
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  	  		  		  		    	 		 		   		 		  
    print("This is LinRegLearner.py. Please use 'LinRegLearner(verbose = True)' to initialize.")
