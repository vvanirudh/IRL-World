"""
Implements the gridworld environment

Derived from Matthew Alger's version of code'

Anirudh Vemula, 2016
avemula1@andrew.cmu.edu
"""
import numpy as np
import numpy.random as rn

from irlworld import *

class Gridworld(IRLworld):
    """
    Gridworld MDP
    """

    def __init__(self, grid_size, wind, discount, four_connectivity=True):
        """
        Initializer function

        grid_size : Grid size (assumed to be square) : int
        wind : Probability of moving randomly : float
        discount : MDP discount factor :  float
        four_connectivity : 4-grid connectivity (default true). If false, then 8-grid connectivity
        """
        if four_connectivity:
            self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        else:
            self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

        self.n_actions = len(self.actions)
        self.grid_size = grid_size
        self.n_states = grid_size**2
        self.wind = wind
        self.discount = discount
        self.four_connectivity = four_connectivity

        # Construct the transition probability array
        self.transition_probability = np.array(
            [[[self._transition_probability(i,j,k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def __str__(self):
        """
        Printing function
        """
        return "Gridworld({}, {}, {}, {})".format(self.grid_size, self.wind, self.discount, self.four_connectivity)

    def feature_vector(self, i, feature_map="ident"):
        """
        Return feature vector of the state specified by the index i

        i : State index
        feature_map : Feature map to use (default ident). Must be one of {ident, coord, proxi}
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f

        elif feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x-a) + abs(y-b)
                    f[self.state_to_index((a,b))] = dist
            return f

        # Assume identity map
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
        Return the feature matrix for the entire gridworld

        feature_map : Feature map to use (default ident). Must be one of {ident, coord, proxi}
        """
        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def index_to_state(self, i):
        """
        Return the grid coordinates corresponding to the given state index

        i : State index
        """
        return (i % self.grid_size, i // self.grid_size)

    def state_to_index(self, p):
        """
        Returns the state index corresponding to the given grid coordinates

        p : A tuple containing the grid coordinates (x,y)        
        """
        return p[0] + p[1]*self.grid_size

    def neighboring(self, i, k):
        """
        Returns if the two states i and k (given by coordinates) are neighbors. Also,
        returns true if they are the same point.

        i : (x,y) int tuple
        k : (x,y) int tuple
        """
        if self.four_connectivity:
            return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1
        else:
            return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 2

    def _transition_probability(self, i, j, k):
        """
        Returns the probability of transitioning from state i to state k given
        action j

        i : State int
        j : Action int
        k : State int
        """
        
        xi, yi = self.index_to_state(i)
        xj, yj = self.actions[j]
        xk, yk = self.index_to_state(k)
        