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

        if not self.neighboring((xi, yi), (xk, yk)):
            return 0

        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # Both are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0,0), (self.grid_size - 1, self.grid_size - 1)
                        (0, self.grid_size - 1), (self.grid_size - 1, 0)}:
            # Corner
            # Was the action intended to move it off the grid
            if not (0 <= xi + xj < self.grid_size and 0 <= yi+yj < self.grid_size):
                # Yes
                # 2 because we can go off-grid in two ways 
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # No
                return 2*self.wind/self.n_actions
        else:
            # Not a Corner
            # An edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}):
                # No
                return 0
            else:
                # Yes
                # Was the action intended to move it off the grid
                if not (0 <= xi+xj < self.grid_size and 0<= yi+yj < self.grid_size):
                    # Yes
                    return 1 - self.wind + self.wind/self.n_actions
                else:
                    # No
                    return self.wind/self.n_actions
        return 0

    def reward(self, i):
        """
        Reward for being in state specified by the given index

        i : State index. int
        """
        if state_int == self.n_states - 1:
            return 1
        else:
            return 0

    def average_reward(self, n_traj, traj_length, policy):
        """
        Returns the average reward obtained by following a given policy over
        n_traj trajectories of length traj_length

        policy : Map from state indices to action indices
        n_traj : Number of trajectories. int
        traj_length : Length of an episode. int
        """

        trajectories = self.generate_trajectories(n_traj, traj_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        total_reward = rewards.sum(axis=1)

        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, i):
        """
        Defines the optimal policy for this gridworld

        i : State index. int
        """
        sx, sy = self.index_to_state(i)

        if sx < self.grid_size and sy < self.grid_size:
            return rn.randint(0,2)
        if sx < self.grid_size-1:
            return 0
        if sy < self.grid_size-1:
            return 1
        raise ValueError("Unexpected state")

    def optimal_policy_deterministic(self, i):
        """
        Deterministic version of the optimal policy for this gridworld

        i : State index. int
        """
        sx, sy = self.index_to_state(i)
        if sx < sy:
            return 0
        else:
            return 1
            
    def generate_trajectories(self, n_traj, traj_length, policy, random_start=False):
        """
        Generates n_traj trajectories with length traj_length according to the given policy

        n_traj : Number of trajectories. int
        traj_length : Length of an episode. int
        policy : Map from state indices to action indices
        random_start : Whether to start randomly (default False). bool
        """

        trajectories = []
        for _ in range(n_traj):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            else:
                sx, sy = 0, 0

            trajectory = []
            for _ in range(traj_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, self.n_actions)]
                else:
                    action = self.actions[policy(self, state_to_index((sx,sy)))]

                if ( 0<= sx + action[0] < self.grid_size and
                     0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.state_to_index((sx,sy))
                action_int = self.actions.index(action)
                next_state_int = self.state_to_index((next_sx, next_sy))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy
                
            trajectories.append(trajectory)

        return np.array(trajectories)            