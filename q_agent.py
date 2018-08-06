"""deep q learning algorith created with keras"""
from collections import deque

import random

import pickle
import numpy as np
import os.path
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding, concatenate
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K


class QqAgent:
    """Deep Q Learning Network with target model and replay learning"""
    def __init__(self, state_size_one, state_matrix_enemies_size, action_size):
        self.weight_backup      = "backup_v1.h5"
        self.action_size        = action_size
        self.memory             = deque(maxlen=16000)
        self.memory_episode     = deque()
        self.learning_rate      = 0.01
        self.gamma              = 0.98
        self.exploration_min    = 0.01

        self.exploration_decay  = 0.995
        if(not self.try_load_model()):
            self.exploration_rate   = 1.0
            self.brain              = self._build_model(state_size_one,state_matrix_enemies_size, action_size)
            # "hack" implemented by DeepMind to improve convergence
            self.target_model       = self._build_model(state_size_one,state_matrix_enemies_size, action_size)

    def _build_model(self, state_size, shape_enemy_map, action_size):

        #data collected from pysc2 is inserted here
        input_one = Input(shape=(state_size,), name='input_one')

        #map of enemies
        input_enemies = Input(shape=shape_enemy_map, name='input_enemies')     	
        conv_one = Conv2D(4, (3, 3), activation='relu', input_shape=(8,8,1))(input_enemies)
        conv_two = Conv2D(6, (3, 3), activation='relu')(conv_one)
        conv_three = Conv2D(8, (3, 3), activation='relu')(conv_two)
        out_conv = Flatten()(conv_three)

        merged = concatenate([input_one, out_conv])
        output_one = Dense(36, activation='relu')(merged)
        output_two = Dense(50, activation='relu')(output_one)
        output_three = Dense(action_size, activation='relu')(output_two)

        model = Model([input_one, input_enemies], output_three)

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    def act(self, state, excluded_actions):
        """
        ask the model what to do
        excluded_actions: this actions will be excluded from the valid actions
        """

        if state is None or np.random.rand() <= self.exploration_rate:
            allowed_actions = []
            for i in range(self.action_size):
                allowed_actions.append(i)
            allowed_actions = np.array(allowed_actions)
            allowed_actions = np.delete(allowed_actions, excluded_actions)
            rand_action = random.randrange(len(allowed_actions))

            return allowed_actions[rand_action]
        #act_values = self.brain.predict(np.reshape(state, [1, len(state)]))[0]
        
        act_values = self.brain.predict([np.reshape(state["state_others"], [1, len(state["state_others"])]), np.reshape(state["state_enemy_matrix"], (1, 8, 8, 1))])

        act_values = self.minimize_excluded(act_values, excluded_actions)

        return np.argmax(act_values)


  
    def target_train(self):
        """copy weights to target network"""
        weights = self.brain.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)    



    def replay(self, sample_batch_size):
        """data replay to train the model"""
        self.memory.extend(self.memory_episode)
        self.memory_episode = deque()

        if len(self.memory) < sample_batch_size:
            return

        minibatch = random.sample(self.memory, sample_batch_size)

        input_others = []
        input_enemy_matrix = []
        targets = []  #32, 2

        for state_t, action_t, reward_t, state_t1, terminal, disallowed_actions in minibatch:
            input_others.append(state_t["state_others"])
            input_enemy_matrix.append(state_t["state_enemy_matrix"])
            
            expected_future_rewards = self.target_model.predict([np.reshape(state_t1["state_others"], [1, len(state_t1["state_others"])]), np.reshape(state_t1["state_enemy_matrix"], (1, 8, 8, 1))])[0]
            expected_future_rewards = self.minimize_excluded(expected_future_rewards, disallowed_actions)

            #exclude invalid actions
            target_prediction = self.target_model.predict([np.reshape(state_t["state_others"], [1, len(state_t["state_others"])]), np.reshape(state_t1["state_enemy_matrix"], (1, 8, 8, 1))])[0]

            if terminal:
                target_prediction[action_t] = reward_t
            else:
                target_prediction[action_t] = (reward_t + self.gamma * np.max(expected_future_rewards))

            targets.append(target_prediction)
        
        self.brain.train_on_batch([input_others, input_enemy_matrix], np.array(targets))
        

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
            print(str(self.exploration_rate))

    def minimize_excluded(self, predictions, excluded_indexes):
        """minimize the indexes from excluded_indexes, by making them small they are ignored"""
        min_value = np.amin(predictions) - 1
        for i in range(len(predictions)):
            if(i in excluded_indexes):
                predictions[i] = min_value
        return predictions

    def try_load_model(self):
        """
        load everything important related to the model
        returns false if model not loaded

        """
        if not os.path.isfile('model/model.h5'):
            return False 
        self.brain = load_model('model/model.h5')
        self.target_model = load_model('model/model.h5')
        with open('model/exploration_rate.p', 'rb') as fp:
            self.exploration_rate = pickle.load(fp)
        return True

    def save_model(self):
        """save everything important related to the model"""

        self.target_model.save('model/model.h5')

        with open('model/exploration_rate.p', 'wb') as fp:
            pickle.dump(self.exploration_rate, fp, protocol=pickle.HIGHEST_PROTOCOL)

