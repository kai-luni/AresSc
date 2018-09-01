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

class AresDdqnNet:
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

        self.tensor_board =  TensorBoard(log_dir="logs/{}".format(time()))
        self.tensor_counter = 0
        self.counter_trained_pictures = 0

        self.memory = self.load_one_super_episode()

        if(not self.try_load_model()):
            self.exploration_rate   = 1.0
            self.brain              = self.build_model(state_size_one,state_matrix_enemies_size, action_size)
            # "hack" implemented by DeepMind to improve convergence
            self.target_model       = self.build_model(state_size_one,state_matrix_enemies_size, action_size)

        self.tensor_board.set_model(self.brain)

    def act(self, state, excluded_actions):
        """
        ask the model what to do
        excluded_actions: this actions will be excluded from the valid actions
        """

        if state is None or np.random.rand() <= self.exploration_rate  or len(self.memory) < 20000:
            allowed_actions = []
            for i in range(self.action_size):
                allowed_actions.append(i)
            allowed_actions = np.array(allowed_actions)
            allowed_actions = np.delete(allowed_actions, excluded_actions)
            rand_action = random.randrange(len(allowed_actions))

            return allowed_actions[rand_action]
        #act_values = self.brain.predict(np.reshape(state, [1, len(state)]))[0]
        
        act_values = self.brain.predict([np.reshape(state["state_others"], [1, len(state["state_others"])]), np.reshape(state["state_enemy_matrix"], (1, 64, 64, 3))])[0]
        act_values = self.minimize_excluded_list([act_values], [excluded_actions])

        return np.argmax(act_values)



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

    def load_one_super_episode(self):
        foldername_episodes_super = "super_episodes"
        onlyfiles = [f for f in listdir(foldername_episodes_super) if isfile(join(foldername_episodes_super, f))]
        if(len(onlyfiles) is 0):
            return[]
        chosen_file_index = random.randint(0, len(onlyfiles)-1)
        filename = onlyfiles[chosen_file_index]
        memory_episode = pickle.load(open(join(foldername_episodes_super, filename), 'rb'))
        return memory_episode

    def minimize_excluded_list(self, predictions_list, excluded_indexes_list):
        """minimize the indexes from excluded_indexes => by making them small they are ignored"""

        return_list = []

        for h in range(len(predictions_list)):
            predictions = predictions_list[h]
            excluded_indexes = excluded_indexes_list[h]

            min_value = np.amin(predictions) - 1
            if(len(predictions) < 4):
                raise ValueError('wrong form of numpy array prediction.')
            for i in range(len(predictions)):
                if(i in excluded_indexes):
                    predictions[i] = min_value
            return_list.append(predictions)
        return return_list

    def save_model(self):
        """save everything important related to the model"""

        self.target_model.save('model/model.h5')

        with open('model/exploration_rate.p', 'wb') as fp:
            pickle.dump(self.exploration_rate, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def target_train(self):
        """copy weights to target network"""
        weights = self.brain.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights) 

    # pick samples randomly from replay memory (with batch_size)
    def replay(self, sample_batch_size, game_score, episode):
        # if len(self.memory) < self.train_start:
        #     return
        if self.exploration_rate > self.exploration_min and len(self.memory) > 20000:
            self.exploration_rate *= self.exploration_decay
            print(str(self.exploration_rate))

        """data replay to train the model"""
        self.memory.extend(self.memory_episode)

        print("samples: " + str(len(self.memory)))
        if len(self.memory) < sample_batch_size:
            sample_batch_size = len(self.memory)
        mini_batch = random.sample(self.memory, sample_batch_size)
        temp_deque = deque()
        for i in range(16):
            temp_deque.extend(self.load_one_super_episode())
        mini_batch.extend(random.sample(temp_deque, int(len(temp_deque)/4)))
        print("mini_batch: " + str(len(mini_batch)))

        # history = np.zeros((len(mini_batch), 64, 64, 3))
        # next_history = np.zeros((len(mini_batch), 64, 64, 3))
        # target = np.zeros((len(mini_batch), ))
        # action, reward, dead = [], [], []

        # for i in range(len(mini_batch)):
        #     history[i] = np.float32(mini_batch[i][0] / 255.)
        #     next_history[i] = np.float32(mini_batch[i][3] / 255.)
        #     action.append(mini_batch[i][1])
        #     reward.append(mini_batch[i][2])
        #     dead.append(mini_batch[i][4])

        history_picture, history_other = [], []
        next_history_picture, next_history_other = [], []
        action, reward, dead = [], [], []
        disallowed_actions_list = []
        for state_t, action_t, reward_t, state_t1, terminal, disallowed_actions in mini_batch:
            history_picture.append(state_t["state_enemy_matrix"])
            history_other.append(state_t["state_others"])
            next_history_picture.append(state_t1["state_enemy_matrix"])
            next_history_other.append(state_t1["state_others"])
            action.append(action_t)
            reward.append(reward_t)
            dead.append(terminal)
            disallowed_actions_list.append(disallowed_actions)

        #history = np.array(history)
        # next_history_picture = np.array(next_history_picture)
        # next_history_other = np.array(next_history_other)
        action = np.array(action)
        reward = np.array(reward)
        dead = np.array(dead)

        value = self.brain.predict([next_history_other, next_history_picture])
        value = self.minimize_excluded_list(value, disallowed_actions_list)
        target_value = self.target_model.predict([next_history_other, next_history_picture])
        target_value = self.minimize_excluded_list(target_value, disallowed_actions_list)
        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(len(mini_batch)):
            if dead[i]:
                target_value[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                target_value[i][action[i]] = reward[i] + self.gamma * target_value[i][np.argmax(value[i])]
        finished = False
        input_others_temp, input_enemy_matrix_temp, targets_temp = [], [], []
        for i in range(len(target_value)):
            input_others_temp.append(history_other[i])
            input_enemy_matrix_temp.append(history_picture[i])
            targets_temp.append(target_value[i])
            if i == (len(target_value) - 1):
                training_loss = self.brain.train_on_batch([input_others_temp, input_enemy_matrix_temp], np.array(targets_temp))
                self.write_plot(episode, [training_loss], game_score, self.memory_episode)
                continue
            if len(targets_temp) % 8000 == 0:
                training_loss = self.brain.train_on_batch([input_others_temp, input_enemy_matrix_temp], np.array(targets_temp))
                input_others_temp, input_enemy_matrix_temp, targets_temp = [], [], []

        #return_fit = self.brain.fit([history_other, history_picture], np.array(target_value), verbose=1, epochs=2)
        #training_loss = self.brain.train_on_batch([input_others, input_enemy_matrix], np.array(targets))
        #training_loss = return_fit.history["loss"]

        
        self.memory_episode = deque()



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