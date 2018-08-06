
"""
deep q learning algorith created with keras
"""
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

class q_agent_two:
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

    """"""
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