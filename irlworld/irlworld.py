"""
Implements the interface for a generic IRL world

Anirudh Vemula, 2016
avemula1@andrew.cmu.edu
"""


class IRLworld(object):
    """
    IRL world MDP
    """
    
    def __init__(self):
        """
        Initializer function
        """
        pass

    def __str__(self):
        """
        Printing function
        """
        pass

    def feature_vector(self, index):
        """
        Returns feature vector for a given state as specified by its index
        """
        pass

    def feature_matrix(self):
        """
        Returns the feature matrix for the entire world
        """
        pass

    def index_to_state(self, index):
        """
        Returns the state for the given index
        """
        pass

    def state_to_index(self, state):
        """
        Returns the index for the given state
        """
        pass

    def _transition_probability(self, i, j, k):
        """
        Returns the probability of transitioning from state index i to state index j
        by taking action index j
        """
        pass

    def reward(self, index):
        """
        Returns the reward for being in state specified by the given index
        """
        pass

    def reward(self, state_index, action_index):
        """
        Returns the reward for taking action specified by action_index
        in state specified by state_index
        """
        pass

    def average_reward(self, num_traj, traj_length, policy):
        """
        Returns the mean and std dev of reward obtained over num_traj trajectories
        of length traj_length according to the given policy
        """
        pass

    def optimal_policy(self, index):
        """
        Encodes the optimal stochastic policy. Returns the optimal action for the given state specified by index
        """
        pass

    def optimal_policy_deterministic(self, index):
        """
        Encodes the optimal deterministic policy. Returns the optimal action for the given state specified by index
        """
        pass

    def generate_trajectories(self, num_traj, traj_length, policy, random_start=False):
        """
        Generates num_traj trajectories in the world of length traj_length according to the given policy
        with random starts or not as specified        
        """
        pass
