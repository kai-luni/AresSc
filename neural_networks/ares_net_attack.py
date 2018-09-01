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

class AresNetAttack():
    """Deep Q Learning Network with target model and replay learning"""
    def __init__(self, state_size_one, state_matrix_enemies_size, action_size):
        self.weight_backup      = "backup_v1.h5"
        self.action_size        = action_size
        self.memory             = deque(maxlen=400000)
        self.memory_episode     = deque()
        self.learning_rate      = 0.008
        self.gamma              = 0.992
        self.exploration_min    = 0.1
        self.exploration_decay  = 0.995

        self.counter_trained_pictures = 0

        self.memory = self.load_one_super_episode_alpha_one()

        if(not self.try_load_model()):
            self.exploration_rate   = 1.0
            self.brain              = self.build_model(state_size_one,state_matrix_enemies_size, action_size)
            # "hack" implemented by DeepMind to improve convergence
            self.target_model       = self.build_model(state_size_one,state_matrix_enemies_size, action_size)


    def build_model(self, state_size, shape_enemy_map, action_size):

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

    def act(self, state):
        """
        ask the model what to do
        """

        if state is None or np.random.rand() <= self.exploration_rate:
            rand_action = random.randrange(len(self.action_size))

            return rand_action
        
        act_values = self.brain.predict([np.reshape(state["state_others"], [1, len(state["state_others"])]), np.reshape(state["state_enemy_matrix"], (1, 64, 64, 3))])[0]

        return np.argmax(act_values)



    def target_train(self):
        """copy weights to target network"""
        weights = self.brain.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights) 


    def load_one_super_episode_alpha_one(self):
        """transform alpha one replays to alpha 2: remove first 4 moves"""
        foldername_episodes_super = "super_episodes_alpha_one"
        onlyfiles = [f for f in listdir(foldername_episodes_super) if isfile(join(foldername_episodes_super, f))]
        if(len(onlyfiles) is 0):
            return[]
        chosen_file_index = random.randint(0, len(onlyfiles)-1)
        filename = onlyfiles[chosen_file_index]
        memory_episode = pickle.load(open(join(foldername_episodes_super, filename), 'rb'))

        return_list = deque()
        for state_t, action_t, reward_t, state_t1, terminal in memory_episode:
            if action_t < 4:
                continue
            state_object = [state_t, action_t-4, reward_t, state_t1, terminal]
            return_list.append(state_object)
        return return_list


    # pick samples randomly from replay memory (with batch_size)
    def replay(self, sample_batch_size, game_score, episode):
        # if len(self.memory) < self.train_start:
        #     return
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
            print(str(self.exploration_rate))

        """data replay to train the model"""
        self.memory.extend(self.memory_episode)

        print("samples: " + str(len(self.memory)))
        if len(self.memory) < sample_batch_size:
            sample_batch_size = int(len(self.memory) / 2)
        mini_batch = random.sample(self.memory, sample_batch_size)
        temp_deque = deque()
        for i in range(16):
            temp_deque.extend(self.load_one_super_episode_alpha_one())
        mini_batch.extend(random.sample(temp_deque, int(len(temp_deque)/4)))
        print("mini_batch: " + str(len(mini_batch)))

        fit_return, _ = self.create_ddqn_data(mini_batch, self.brain, self.target_model, self.gamma)
        training_loss = fit_return.history["loss"]
        self.write_plot(episode, training_loss, game_score, self.memory_episode)
        self.memory_episode = deque()


    @staticmethod
    def create_ddqn_data(batch, brain, target_model, gamma):
        history_picture, history_other = [], []
        next_history_picture, next_history_other = [], []
        action, reward, dead = [], [], []
        for state_t, action_t, reward_t, state_t1, terminal in batch:
            history_picture.append(state_t["state_enemy_matrix"])
            history_other.append(state_t["state_others"])
            next_history_picture.append(state_t1["state_enemy_matrix"])
            next_history_other.append(state_t1["state_others"])
            action.append(action_t)
            reward.append(reward_t)
            dead.append(terminal)

        action = np.array(action)
        reward = np.array(reward)
        dead = np.array(dead)

        value = brain.predict([next_history_other, next_history_picture])
        target_value = target_model.predict([next_history_other, next_history_picture])
        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(len(batch)):
            if dead[i]:
                target_value[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                target_value[i][action[i]] = reward[i] + gamma * target_value[i][np.argmax(value[i])]


        return_fit = brain.fit([history_other, history_picture], np.array(target_value), verbose=1, epochs=2)
        

        return return_fit, brain


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