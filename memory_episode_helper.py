from rl.memory import Memory, RingBuffer
from time import time
from os import listdir
from os.path import isfile, join

import pickle
import random
import sys
import os

from ares_processor import AresProcessor

class MemoryEpisodeHelper(Memory):
    """Handle custom memory activities, episode related."""
    def __init__(self):
        self.limit = 1000000
        self.foldername_episodes_super = "super_episodes"
        self.version = "alpha_3"

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.observations = []

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory
        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """ 
        #super(MemoryEpisodeHelper, self).append(observation, action, reward, terminal, training=training)
        
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def save_episode(self, game_score):
        """store the whole episode in a pickle file with a unique name"""
        if game_score < 8000: 
            return
        memory_episode = {}
        memory_episode["actions"] = self.actions
        memory_episode["rewards"] = self.rewards
        memory_episode["terminals"] = self.terminals
        memory_episode["observations"] = self.observations
        # for i in range(len(memory_episode["terminals"])):
        #     print(str(memory_episode["terminals"][i]))
        assert memory_episode["terminals"][len(memory_episode["terminals"])-1]
        folder_name = self.foldername_episodes_super + '_' + self.version
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        pickle.dump(memory_episode, open(folder_name + '/' + str(game_score) + '_' + str(time()) + '.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL) 

    def load_random_episode_into_other_memory(self, memory):
        try:
            folder_name = self.foldername_episodes_super + '_' + self.version
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
            if(len(onlyfiles) is 0):
                return[]
            chosen_file_index = random.randint(0, len(onlyfiles)-1)
            filename = onlyfiles[chosen_file_index]
            memory_episode = pickle.load(open(join(folder_name, filename), 'rb'))
            assert memory_episode["terminals"][len(memory_episode["terminals"])-1]
            for i in range(len(memory_episode["actions"])):
                memory.append(memory_episode["observations"][i], memory_episode["actions"][i], memory_episode["rewards"][i], memory_episode["terminals"][i], True)
            return True
        except Exception as e:
            print("error loading super episode: " + str(e))
            return False
