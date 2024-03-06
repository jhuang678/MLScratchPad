import numpy as np

class BagLearner(object):

    def __init__(self, learner:object, kwargs:dict, bags:int, boost:bool = False, verbose:bool = False):
        self.learners = []
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        for i in range(0, bags):
            self.learners.append(learner(**kwargs))

        if self.verbose:
            print("\nInitialized Bootstrap Aggregation:")
            print("Author: ", self.author())
            print("Learner:", learner)
            print("Keyword Arguments:", kwargs)
            print("Number of Bags: ", self.bags)
            print("Boost: ", self.boost)

    def author(self):
        """
        :rtype: str  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        return "jhuang678"

    def add_evidence(self, data_x:float, data_y:float):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		  	  		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		  	  		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		  	  		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        for learner in self.learners:
            index_rand = np.random.choice(range(data_x.shape[0]), data_x.shape[0], replace = True)
            new_data_x = data_x[index_rand]
            new_data_y = data_y[index_rand]
            learner.add_evidence(new_data_x, new_data_y)

        if self.verbose:
            print("Please use 'query(points)' to predict values.")

    def query(self, points:float):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		  	  		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  	  		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        return sum([learner.query(points) for learner in self.learners]) / len(self.learners)

if __name__ == "__main__":
    print("This is BagLearner.py. Please use 'BagLearner(learner, kwargs, bags, boost,  verbose)' to initialize.")
