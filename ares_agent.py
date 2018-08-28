"""Main Module which is executed by the Pysc2 library"""

import random
import math
import os.path
import skimage
import pickle
import cv2

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import actions, features, units

from neural_networks.ares_ddqn_net import AresDdqnNet
from map_matrix import get_eight_by_eight_matrix
from point_rect import Point
from reward.reward_calculator import RewardCalculator
from helper_functions.normalizer import normalize
from helper_functions.obs_helper import get_random_unit, get_count_unit
from skimage.color import rgb2gray

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'sparse_agent_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
]

MAP_MATRIX = get_eight_by_eight_matrix(64, 64)

for height in range(8):
    for width in range(8):
        center_point = MAP_MATRIX[height][width].get_center()
        smart_actions.append(ACTION_ATTACK + '_' + str(center_point.x) + '_' + str(center_point.y))

class AresAgent(base_agent.BaseAgent):
    def __init__(self):
        super(AresAgent, self).__init__()

        self.reward_calc = RewardCalculator()

        action_size = len(smart_actions)
        state_size = 4
        self.qlearn = AresDdqnNet(state_size_one=state_size, state_matrix_enemies_size=(8, 8, 1), action_size=action_size)
        self.steps_last_learn = 0


        self.previous_action = None
        self.previous_state = None
        self.previous_obs = None
        self.last_score = 0
        
        self.cc_y = None
        self.cc_x = None
        
        self.move_number = 0

        self.last_killed_unit_score = 0
        self.Last_killed_building_score = 0

        
    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        
        return [x + x_distance, y + y_distance]
    
    def transformLocation(self, x, y):
        # if not self.base_top_left:
        #     return [64 - x, 64 - y]
        
        return [x, y]
    
    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
            
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def reset(self):
        """called between 2 episodes"""
        super(AresAgent, self).reset()
        if(self.episodes%3 == 0):
            self.qlearn.target_train()
        self.qlearn.save_model()
        
    def step(self, obs):
        super(AresAgent, self).step(obs)

        unit_type = obs.observation['rgb_screen'][_UNIT_TYPE]

        if obs.first():
            if(self.episodes < 2):
                self.episodes = 1 if not os.path.isfile('model/episodes.p') else pickle.load(open('model/episodes.p', 'rb'))[len(pickle.load(open('model/episodes.p', 'rb')))-1]

            player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()
      
            if xmean <= 31 and ymean <= 31:
                self.base_top_left = 1
            else:
                self.base_top_left = 0

        
        commCenter = get_random_unit(obs, units.Terran.CommandCenter)
        if(commCenter is not None):
            self.cc_x = commCenter.x
            self.cc_y = commCenter.y
        
        supply_depot_count = get_count_unit(obs, units.Terran.SupplyDepot)
        barracks_count = get_count_unit(obs, units.Terran.Barracks)

        supply_used = obs.observation['player'][3]
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        worker_supply = obs.observation['player'][6]
        
        supply_free = supply_limit - supply_used

        excluded_actions = []
        if supply_depot_count == 2 or worker_supply == 0:
            excluded_actions.append(1)
            
        if supply_depot_count == 0 or barracks_count == 2 or worker_supply == 0:
            excluded_actions.append(2)

        #TODO: when building is not finished building its counted here already
        if supply_free == 0 or barracks_count == 0:
            excluded_actions.append(3)
            
        if army_supply == 0:
            for i in range(68):
                if(i > 3):
                    excluded_actions.append(i)        


        if obs.last():
            self.qlearn.replay(30000, obs.observation["score_cumulative"][0], self.episodes)

            current_state = self.getCurrentState(obs)

            stateObject = [self.previous_state, self.previous_action, obs.reward, current_state, obs.last(), excluded_actions]

            self.qlearn.memory_episode.append(stateObject)

            
            self.previous_action = None
            self.previous_state = None
            self.previous_obs = None
            self.last_score = 0
            
            self.move_number = 0

            self.last_killed_unit_score = 0
            self.Last_killed_building_score = 0
            
            return actions.FunctionCall(_NO_OP, [])
            
        if self.move_number == 0:
            self.move_number += 1
            value = self.moveNumberZero(obs, unit_type, excluded_actions)
            if(value[0] != 0):
                return value
            return value
        
        elif self.move_number == 1:
            self.move_number += 1
            value =  self.moveNumberOne(obs)
            return value
                
        elif self.move_number == 2:
            self.move_number = 0
            value =  self.moveNumberTwo(obs, unit_type)
            return value

        return actions.FunctionCall(_NO_OP, [])

    def moveNumberZero(self, obs, unit_type, excluded_actions):
        # supply_depot_count = get_count_unit(obs, units.Terran.SupplyDepot)
        # barracks_count = get_count_unit(obs, units.Terran.SupplyDepot)        
        current_state = self.getCurrentState(obs)

        if self.previous_action is not None and not obs.last():
            killed_unit_score = obs.observation['score_cumulative'][5]
            killed_building_score = obs.observation['score_cumulative'][6]
            #rewardDiff = obs.observation['score_cumulative'][0] - self.last_score            
            
            #if(self.last_score is 0):
                #reward = 0
            self.last_score = obs.observation['score_cumulative'][0]
            reward = self.reward_calc.get_reward_from_observation(self.previous_obs, self.previous_action)
            state_object = [self.previous_state, self.previous_action, reward, current_state, obs.last(), excluded_actions]
            #state_object = self.calculate_reward(state_object, self.last_killed_unit_score, killed_unit_score, self.Last_killed_building_score, killed_building_score)
            self.last_killed_unit_score = killed_unit_score
            self.Last_killed_building_score = killed_building_score

            self.qlearn.memory_episode.append(state_object)

            # self.steps_last_learn +=1
            # if(self.steps_last_learn > 400):
            #     self.qlearn.replay(1000)
            #     self.steps_last_learn = 0
            #self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))
    
        #rl_action = self.qlearn.choose_action(str(current_state))
        rl_action = self.qlearn.act(current_state, excluded_actions)
        self.previous_state = current_state
        self.previous_obs = obs
        self.previous_action = rl_action
    
        smart_action, _, _ = self.splitAction(self.previous_action)
        
        if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            scv = get_random_unit(obs, units.Terran.SCV)

            if(scv is not None):
                return actions.FUNCTIONS.select_point("select", (scv.x, scv.y))
            # unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                
            # if unit_y.any():
            #     i = random.randint(0, len(unit_y) - 1)
            #     target = [unit_x[i], unit_y[i]]
                

            #     return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            
        elif smart_action == ACTION_BUILD_MARINE:
            barrack = get_random_unit(obs, units.Terran.Barracks)
            if(barrack is not None):
                return actions.FUNCTIONS.select_point("select_all_type", (barrack.x, barrack.y))
            # if barracks_y.any():
            #     i = random.randint(0, len(barracks_y) - 1)
            #     target = [barracks_x[i], barracks_y[i]]
            #     if(barracks_x[i] < 0 or barracks_y[i] < 0):
            #         print("oh no")
            #     return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
            
        elif smart_action == ACTION_ATTACK:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        return actions.FunctionCall(_NO_OP, [])

    def moveNumberOne(self, obs):
        smart_action, x, y = self.splitAction(self.previous_action)
        supply_depot_count = get_count_unit(obs, units.Terran.SupplyDepot)
        barracks_count = get_count_unit(obs, units.Terran.Barracks)
            
        if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if (actions.FUNCTIONS.Build_SupplyDepot_screen.id in obs.observation.available_actions):
                if self.cc_y.any():
                    if supply_depot_count == 0:
                        target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)
                    elif supply_depot_count == 1:
                        target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)
                        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)

                    

        
        elif smart_action == ACTION_BUILD_BARRACKS:

            if barracks_count < 2 and actions.FUNCTIONS.Build_Barracks_screen.id in obs.observation.available_actions:
                if self.cc_y.any():
                    if  barracks_count == 0:
                        target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                    elif  barracks_count == 1:
                        target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)

                    return actions.FUNCTIONS.Build_Barracks_screen("now", target)

        elif smart_action == ACTION_BUILD_MARINE:
            if _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
    
        elif smart_action == ACTION_ATTACK:
            do_it = True
            
            if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                do_it = False
            
            if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                do_it = False
            
            if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                x_offset = random.randint(-1, 1)
                y_offset = random.randint(-1, 1)
                
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(float(x) + (x_offset * 2), float(y) + (y_offset * 2))])
        return actions.FunctionCall(_NO_OP, [])

    def moveNumberTwo(self, obs, unit_type):
        smart_action, _, _ = self.splitAction(self.previous_action)
            
        if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if (actions.FUNCTIONS.Harvest_Gather_screen.id in obs.observation.available_actions):
                mineral_field = get_random_unit(obs, units.Neutral.MineralField)
                if mineral_field is not None:
                    return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, [mineral_field.x, mineral_field.y]])
            #if _HARVEST_GATHER in obs.observation['available_actions']:
                #unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
                
                # if unit_y.any():
                #     i = random.randint(0, len(unit_y) - 1)
                    
                #     m_x = unit_x[i]
                #     m_y = unit_y[i]
                    
                #     target = [int(m_x), int(m_y)]
                    
                #     return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
        return actions.FunctionCall(_NO_OP, [])

    def getCurrentState(self, obs):
        supply_depot_count = get_count_unit(obs, units.Terran.SupplyDepot)
        barracks_count = get_count_unit(obs, units.Terran.Barracks)
        current_state = []
        current_state.append(normalize(get_count_unit(obs, units.Terran.CommandCenter), 0, 1))
        current_state.append(normalize(supply_depot_count, 0, 2))
        current_state.append(normalize(barracks_count, 0, 2))
        army_supply = obs.observation['player'][_ARMY_SUPPLY]
        current_state.append(normalize(army_supply, 0, 19))

        #map_matrix_enemy = get_eight_by_eight_matrix(64, 64)
      
        # enemy_y, enemy_x = (obs.observation['rgb_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        # for i in range(0, len(enemy_y)):
        #     enemy_position = Point(enemy_x[i] , enemy_y[i])
        #     for height in range(8):
        #         for width in range(8):
        #             if(map_matrix_enemy[height][width].contains(enemy_position)):
        #                 map_matrix_enemy[height][width].value += 1
        #                 break

        # #TODO own object
        # for height in range(8):
        #     for width in range(8):
        #         #normalize field to -1 to 1
        #         map_matrix_enemy[height][width] = normalize(map_matrix_enemy[height][width].value, 0, 30)

        #np_array_enemies = np.array(map_matrix_enemy).reshape(8,8,1)

        #test_gray = rgb2gray(skimage.img_as_ubyte(obs.observation['rgb_minimap']))
        #cv2.imwrite('color_img.jpg', obs.observation['rgb_minimap'])
        test_pure = (obs.observation['rgb_minimap'] / 128) - 1
        #test_min = np.min(test_pure)
        #test_max = np.max(test_pure)
        #test_final = (test_gray * 2) - 1

        return_dict = {}
        return_dict["state_enemy_matrix"] = test_pure
        return_dict["state_others"] = np.array(current_state)
        return return_dict




    # def calculate_reward(self, state_object, last_killed_units, killed_units, last_killed_buildings, killed_buildings):
    #     reward = 0
    #     if(killed_units > last_killed_units):
    #         reward += KILL_UNIT_REWARD

    #     if(killed_buildings > last_killed_buildings):
    #         reward += KILL_BUILDING_REWARD

    #     if(state_object[0][3] < 0.4 and state_object[1] != 3):
    #         reward -= BUILD_FIGHTING_UNIT_REWARD

    #     if(reward < -1):
    #         reward = -1
    #     if(reward > 1):
    #         reward = 1
    #     state_object[2] = reward

    #     return state_object




        
