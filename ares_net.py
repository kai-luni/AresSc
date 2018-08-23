"""deep q learning algorith created with keras"""
from collections import deque
from os import listdir
from os.path import isfile, join

import random

import pickle
import numpy as np
import os.path
import tensorflow as tf
from time import time

from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding, concatenate
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K


class AresNet:
    """Deep Q Learning Network with target model and replay learning"""
    def __init__(self, state_size_one, state_matrix_enemies_size, action_size):
        self.weight_backup      = "backup_v1.h5"
        self.action_size        = action_size
        self.memory             = deque(maxlen=300000)
        self.memory_episode     = deque()
        self.learning_rate      = 0.008
        self.gamma              = 0.992
        self.exploration_min    = 0.1
        self.exploration_decay  = 0.992

        self.tensor_board =  TensorBoard(log_dir="logs/{}".format(time()))
        self.tensor_counter = 0
        self.counter_trained_pictures = 0

        self.memory = self.load_super_episodes()

        if(not self.try_load_model()):
            self.exploration_rate   = 1.0
            self.brain              = self._build_model(state_size_one,state_matrix_enemies_size, action_size)
            # "hack" implemented by DeepMind to improve convergence
            self.target_model       = self._build_model(state_size_one,state_matrix_enemies_size, action_size)

        self.tensor_board.set_model(self.brain)

    def _build_model(self, state_size, shape_enemy_map, action_size):

        #data collected from pysc2 is inserted here
        input_one = Input(shape=(state_size,), name='input_one')

        #map of enemies
        input_enemies = Input(shape=(64, 64, 3), name='input_enemies')     	
        conv_one = Conv2D(12, kernel_size = (4, 4) ,strides = (2, 2), activation='relu', input_shape=(64,64,3))(input_enemies)
        conv_two = Conv2D(32, kernel_size = (4, 4) ,strides = (2, 2), activation='relu')(conv_one)
        conv_three = Conv2D(128, kernel_size = (4, 4) ,strides = (3, 3), activation='relu')(conv_two)
        out_conv = Flatten()(conv_three)

        merged = concatenate([input_one, out_conv])
        output_one = Dense(1400, activation='relu')(merged)
        output_two = Dense(500, activation='relu')(output_one)
        output_three = Dense(action_size, activation='relu')(output_two)

        model = Model([input_one, input_enemies], output_three)

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
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
        
        act_values = self.brain.predict([np.reshape(state["state_others"], [1, len(state["state_others"])]), np.reshape(state["state_enemy_matrix"], (1, 64, 64, 3))])[0]
        act_values = self.minimize_excluded(act_values, excluded_actions)

        return np.argmax(act_values)


  
    def target_train(self):
        """copy weights to target network"""
        weights = self.brain.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)    



    def replay(self, sample_batch_size, game_score, episode):


        """data replay to train the model"""
        self.memory.extend(self.memory_episode)
        

        print("samples: " + str(len(self.memory)))
        if len(self.memory) < sample_batch_size:
            sample_batch_size = len(self.memory)

        minibatch = random.sample(self.memory, sample_batch_size)

        input_others = []
        input_enemy_matrix = []
        targets = []  #32, 2

        zero_counter = 0
        summary_actions = {}
        max_counter = len(minibatch)/45
        for state_t, action_t, reward_t, state_t1, terminal, disallowed_actions in minibatch:
            #stop mass learning of one action
            if(action_t in summary_actions.keys() and summary_actions[action_t] > max_counter):
                continue
            if(action_t in summary_actions.keys()):
                summary_actions[action_t] += 1
            else:
                summary_actions[action_t] = 1
            if(summary_actions[action_t] > max_counter):
                continue

            #avoid huge amount of do nothing
            if(action_t == 0):
                zero_counter += 1
            if(action_t == 0 and zero_counter > 10):
                continue

            input_others.append(state_t["state_others"])
            input_enemy_matrix.append(np.reshape(state_t1["state_enemy_matrix"], (64, 64, 3)))
            
            expected_future_rewards = self.target_model.predict([np.reshape(state_t1["state_others"], [1, len(state_t1["state_others"])]), np.reshape(state_t1["state_enemy_matrix"], (1, 64, 64, 3))])[0]
            expected_future_rewards = self.minimize_excluded(expected_future_rewards, disallowed_actions)

            #exclude invalid actions
            target_prediction = self.target_model.predict([np.reshape(state_t["state_others"], [1, len(state_t["state_others"])]), np.reshape(state_t1["state_enemy_matrix"], (1, 64, 64, 3))])[0]

            if terminal:
                target_prediction[action_t] = reward_t
            else:
                target_prediction[action_t] = (reward_t + self.gamma * np.max(expected_future_rewards))

            targets.append(target_prediction)


        summary_actions_string = ""
        for key, value in summary_actions.items():
            summary_actions_string += str(key) + ":" + str(value) + "  "
        print(summary_actions_string)

        self.counter_trained_pictures += len(input_others)
        print("trained pictures: " + str(self.counter_trained_pictures))
        #return_fit = self.brain.fit([input_others, input_enemy_matrix], np.array(targets), verbose=1, epochs=3)
        training_loss = self.brain.train_on_batch([input_others, input_enemy_matrix], np.array(targets))
        #logs = self.brain.test_on_batch([input_others, input_enemy_matrix], np.array(targets))
        #training_loss = return_fit.history["loss"]
        self.write_plot(episode, training_loss, game_score, self.memory_episode)
        self.memory_episode = deque()
        #print("training loss: " + str(training_loss))
        #self.write_log(self.tensor_board, [test, game_score], episode)
        
        
        

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
            print(str(self.exploration_rate))

    def minimize_excluded(self, predictions, excluded_indexes):
        """minimize the indexes from excluded_indexes => by making them small they are ignored"""
        min_value = np.amin(predictions) - 1
        if(len(predictions) < 4):
            raise ValueError('wrong form of numpy array prediction.')
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

    def write_plot(self, episode, loss, game_score, memory_episode):
        if not os.path.isfile('model/episodes.p'):
            pickle.dump([episode], open('model/episodes.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            episodes = pickle.load(open('model/episodes.p', 'rb'))
            episodes.append(episode)
            pickle.dump(episodes, open('model/episodes.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)       

        if not os.path.isfile('model/losses.p'):
            pickle.dump([loss], open('model/losses.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            losses = pickle.load(open('model/losses.p', 'rb'))
            losses.append(loss)
            pickle.dump(losses, open('model/losses.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)      

        if not os.path.isfile('model/game_scores.p'):
            pickle.dump([game_score], open('model/game_scores.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            game_scores = pickle.load(open(join('model', 'game_scores.p'), 'rb'))
            game_scores.append(game_score)
            pickle.dump(game_scores, open(join('model', 'game_scores.p'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)                                   

        if(game_score > 8000):
            foldername_episodes_super = "super_episodes"
            pickle.dump(memory_episode, open(foldername_episodes_super + '/' + str(game_score) + '_' + str(time()) + '.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL) 

    def load_super_episodes(self):
        return_list = []
        foldername_episodes_super = "super_episodes"
        onlyfiles = [f for f in listdir(foldername_episodes_super) if isfile(join(foldername_episodes_super, f))]
        for filename in onlyfiles:
            memory_episode = pickle.load(open(join(foldername_episodes_super, filename), 'rb'))
            return_list.extend(memory_episode)
        return return_list

    def save_model(self):
        """save everything important related to the model"""

        self.target_model.save('model/model.h5')

        with open('model/exploration_rate.p', 'wb') as fp:
            pickle.dump(self.exploration_rate, fp, protocol=pickle.HIGHEST_PROTOCOL)




