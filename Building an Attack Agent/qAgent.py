from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import numpy as np
import random
from collections import deque

from replayMemory import ReplayMemory

class qAgent:
    def __init__(self, state_size, action_size):
        self.weight_backup      = "cartpole_weight.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=100)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.99995
        self.brain              = self._build_model()

    def _build_model(self):
            # Neural Net for Deep-Q learning Model
            model = Sequential()
            model.add(Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            return model

    def act(self, state):
            if state is None or np.random.rand() <= self.exploration_rate:
                return random.randrange(self.action_size)
            act_values = self.brain.predict(np.reshape(state, [1, 20]))
            return np.argmax(act_values[0])

    def replay(self, sample_batch_size):
            if len(self.memory) < sample_batch_size:
                return
            sample_batch = random.sample(self.memory, sample_batch_size)
            for state, action, reward, next_state, done in sample_batch:
                target = reward
                if not done:
                    #self.brain.predict(np.reshape(next_state, [1, 20]))
                    #print(str(next_state.shape))
                    #print(str(next_state))
                    target = reward + self.gamma * np.amax(self.brain.predict(np.reshape(next_state, [1, 20]))[0])
                target_f = self.brain.predict(np.reshape(state, [1, 20]))
                target_f[0][action] = target
                self.brain.fit(np.reshape(state,[1, 20]), target_f, epochs=1, verbose=0)
            if self.exploration_rate > self.exploration_min:
                self.exploration_rate *= self.exploration_decay