from replayMemory import ReplayMemory

import numpy as np
import random

class DoubleDQNAgent:

    def __init__(self, state_size, action_size, trace_length):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 5000
        self.explore = 50000
        self.frame_per_action = 4
        self.trace_length = trace_length
        self.update_target_freq = 3000
        self.timestep_per_train = 5 # Number of timesteps between training interval

        # Create replay memory
        self.memory = ReplayMemory()

        # Create main model and target model
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.stats_window_size= 50 # window size for computing rolling statistics
        self.mavg_score = [] # Moving Average of Survival Time
        self.var_score = [] # Variance of Survival Time
        self.mavg_ammo_left = [] # Moving Average of Ammo used
        self.mavg_kill_counts = [] # Moving Average of Kill Counts

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
  
            # Use all traces for RNN
            #q = self.model.predict(state) # 1x8x3
            #action_idx = np.argmax(q[0][-1])

            # Only use last trace for RNN
            q = self.model.predict(state) # 1x3
            action_idx = np.argmax(q)
        return action_idx

    def shape_reward(self, r_t, misc, prev_misc, t):
        
        # Check any kill count
        if (misc[0] > prev_misc[0]):
            r_t = r_t + 1

        if (misc[1] < prev_misc[1]): # Use ammo
            r_t = r_t - 0.1

        if (misc[2] < prev_misc[2]): # Loss HEALTH
            r_t = r_t - 0.1

        return r_t

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):

        sample_traces = self.memory.sample(self.batch_size, self.trace_length) # 32x8x4

        # Shape (batch_size, trace_length, img_rows, img_cols, color_channels)
        update_input = np.zeros(((self.batch_size,) + self.state_size)) # 32x8x64x64x3
        update_target = np.zeros(((self.batch_size,) + self.state_size))

        action = np.zeros((self.batch_size, self.trace_length)) # 32x8
        reward = np.zeros((self.batch_size, self.trace_length))

        for i in range(self.batch_size):
            for j in range(self.trace_length):
                update_input[i,j,:,:,:] = sample_traces[i][j][0]
                action[i,j] = sample_traces[i][j][1]
                reward[i,j] = sample_traces[i][j][2]
                update_target[i,j,:,:,:] = sample_traces[i][j][3]

        """
        # Use all traces for training
        # Size (batch_size, trace_length, action_size)
        target = self.model.predict(update_input) # 32x8x3
        target_val = self.model.predict(update_target) # 32x8x3

        for i in range(self.batch_size):
            for j in range(self.trace_length):
                a = np.argmax(target_val[i][j])
                target[i][j][int(action[i][j])] = reward[i][j] + self.gamma * (target_val[i][j][a])
        """

        # Only use the last trace for training
        target = self.model.predict(update_input) # 32x3
        target_val = self.model.predict(update_target) # 32x3

        for i in range(self.batch_size):
            a = np.argmax(target_val[i])
            target[i][int(action[i][-1])] = reward[i][-1] + self.gamma * (target_val[i][a])

        loss = self.model.train_on_batch(update_input, target)

        return np.max(target[-1,-1]), loss

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)