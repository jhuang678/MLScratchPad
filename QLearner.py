import random as rand
import numpy as np
class QLearner(object):
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    This is a Q learner object.
    :param num_states: The number of states to consider.
    :type num_states: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		  	  		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    def __init__(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        num_states=100,
        num_actions=4,
        alpha = 0.2,
        gamma = 0.9,
        rar = 0.5,
        radr = 0.99,
        dyna = 0,
        verbose=False,
    ):
        self.verbose = verbose
        self.num_states = num_states #integer, the number of states to consider
        self.num_actions = num_actions #integer, the number of actions available.
        self.alpha = alpha #float, the learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
        self.gamma = gamma #float, the discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
        self.rar = rar #float, random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
        self.radr = radr #float, random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically, 0.99.
        self.dyna = dyna #integer, number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
        self.s = 0
        self.a = 0
        self.Q = np.zeros((num_states, num_actions),dtype = float)
        self.R = np.zeros((num_states, num_actions), dtype=float)
        self.T = np.full((num_states, num_actions, num_states), 1/num_states,dtype = float)
        self.Tc = np.full((num_states, num_actions, num_states), 0.00001, dtype=float)
        self.sa_pair =[]

    def author(self): return "jhuang678"

    def querysetstate(self, s):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table.

        A special version of the query method that sets the state to s, and returns an integer action according to the
        same rules as query() (including choosing a random action sometimes), but it does not execute an update to the
        Q-table. It also does not update rar.

        There are two main uses for this method:
            1) To set the initial state, and
            2) when using a learned policy, but not updating it.

        :param s: The new state
        :type s: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        self.s = s

        action = np.argmax(self.Q[s])

        if self.verbose: print(f"s = {s}, a = {action}")

        self.a = action

        return action

    def querysetstate_fast(self, s):
        return np.argmax(self.Q[s])

    def query(self, s_prime, r):
        """
        Update the Q table and return an action  		  	   		  	  		  		  		    	 		 		   		 		  
        :param s_prime: The new state
        :type s_prime: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		  	  		  		  		    	 		 		   		 		  
        :type r: float  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        # Update Q Table
        s = self.s
        a = self.a
        self.Tc[s, a, s_prime] = self.Tc[s, a, s_prime] + 1
        self.sa_pair.append((s,a))

        self.Q[s,a] = (1-self.alpha) * self.Q[s,a] + self.alpha*(r + self.gamma*self.Q[s_prime,np.argmax(self.Q[s_prime])])

        if (self.dyna > 0):
            self.R[s, a] = (1 - self.alpha) * self.R[s, a] + self.alpha * r
            self.DynaQ()

        if rand.uniform(0.0, 1.0) <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
            self.rar = self.rar * self.radr
        else:
            action = np.argmax(self.Q[s_prime])

        if self.verbose: print(f"s = {s_prime}, a = {action}, r = {r}")

        self.s = s_prime
        self.a = action

        return action

    def DynaQ(self):
        # Update T and R
        self.T = self.Tc/self.Tc.sum()

        # Hallucinate
        for d in range(self.dyna):

            s, a = self.sa_pair[rand.randint(0, len(self.sa_pair)-1)]
            s_prime = np.argmax(self.T[s,a])
            r = self.R[s,a]
            self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime])])

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")



