import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from qAgent import qAgent

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

KILL_UNIT_REWARD = 0.5
KILL_BUILDING_REWARD = 1
BUILD_FIGHTING_UNIT_REWARD = 0.5

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))


class SparseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgent, self).__init__()
        
        action_size = len(smart_actions)
        state_size = 8
        self.qlearn = qAgent(state_size = state_size, action_size = action_size)
        self.steps_last_learn = 0


        self.previous_action = None
        self.previous_state = None
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
        if not self.base_top_left:
            return [64 - x, 64 - y]
        
        return [x, y]
    
    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
            
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)
        
    def step(self, obs):
        super(SparseAgent, self).step(obs)
        
        unit_type = obs.observation['screen'][_UNIT_TYPE]

        

        if obs.first():
            player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0
        
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))

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

        if supply_free == 0 or barracks_count == 0:
            excluded_actions.append(3)
            
        if army_supply == 0:
            excluded_actions.append(4)
            excluded_actions.append(5)
            excluded_actions.append(6)
            excluded_actions.append(7)


        if obs.last():
            # current_state = self.getCurrentState(obs, cc_count, supply_depot_count, barracks_count)

            # reward = 0
            # if(obs.observation['score_cumulative'][0] < 6000):
            #     reward = -1
            # if(obs.observation['score_cumulative'][0] > 8000):
            #     reward = 1

            # stateObject = [self.previous_state, self.previous_action, reward, current_state, obs.last(), excluded_actions]
            # if(not (self.previous_state == current_state).all()):
            #     self.qlearn.memory_episode.append(stateObject)
            # self.qlearn.replayTwo(len(self.qlearn.memory_episode), reward)
            #self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
            
            #self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
            self.qlearn.target_train()
            self.previous_action = None
            self.previous_state = None
            self.last_score = 0
            
            self.move_number = 0

            self.last_killed_unit_score = 0
            self.Last_killed_building_score = 0
            
            return actions.FunctionCall(_NO_OP, [])
            
        if self.move_number == 0:
            self.move_number += 1
            value = self.moveNumberZero(obs, cc_count, supply_depot_count, barracks_count, barracks_x, barracks_y, unit_type, excluded_actions)
            return value
        
        elif self.move_number == 1:
            self.move_number += 1
            value =  self.moveNumberOne(obs, supply_depot_count, barracks_count)
            return value
                
        elif self.move_number == 2:
            self.move_number = 0
            value =  self.moveNumberTwo(obs, unit_type)
            return value
            
        

        
        return actions.FunctionCall(_NO_OP, [])

    def moveNumberZero(self, obs, cc_count, supply_depot_count, barracks_count, barracks_x, barracks_y, unit_type, excluded_actions):
        
        current_state = self.getCurrentState(obs, cc_count, supply_depot_count, barracks_count)

        if self.previous_action is not None and not obs.last():
            killed_unit_score = obs.observation['score_cumulative'][5]
            killed_building_score = obs.observation['score_cumulative'][6]
            #rewardDiff = obs.observation['score_cumulative'][0] - self.last_score            
            
            #if(self.last_score is 0):
                #reward = 0
            self.last_score = obs.observation['score_cumulative'][0]
            #reward = self.normalize(obs.observation['score_cumulative'][0], 0, 12000)
            state_object = [self.previous_state, self.previous_action, 0, current_state, obs.last(), excluded_actions]
            state_object = self.calculate_reward(state_object, self.last_killed_unit_score, killed_unit_score, self.Last_killed_building_score, killed_building_score)
            self.last_killed_unit_score = killed_unit_score
            self.Last_killed_building_score = killed_building_score
            if(not (self.previous_state == current_state).all()):
                self.qlearn.memory_episode.append(state_object)
            self.steps_last_learn +=1
            if(self.steps_last_learn > 400):
                self.qlearn.replayTwo(500)
                self.steps_last_learn = 0
            #self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))
    
        #rl_action = self.qlearn.choose_action(str(current_state))
        rl_action = self.qlearn.act(current_state, excluded_actions)
        self.previous_state = current_state
        self.previous_action = rl_action
    
        smart_action, x, y = self.splitAction(self.previous_action)
        
        if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            
        elif smart_action == ACTION_BUILD_MARINE:
            if barracks_y.any():
                i = random.randint(0, len(barracks_y) - 1)
                target = [barracks_x[i], barracks_y[i]]
        
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
            
        elif smart_action == ACTION_ATTACK:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        return actions.FunctionCall(_NO_OP, [])

    def moveNumberOne(self, obs, supply_depot_count, barracks_count):
        smart_action, x, y = self.splitAction(self.previous_action)
            
        if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if supply_depot_count < 2 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                if self.cc_y.any():
                    if supply_depot_count == 0:
                        target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                    elif supply_depot_count == 1:
                        target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)

                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
        
        elif smart_action == ACTION_BUILD_BARRACKS:
            if barracks_count < 2 and _BUILD_BARRACKS in obs.observation['available_actions']:
                if self.cc_y.any():
                    if  barracks_count == 0:
                        target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                    elif  barracks_count == 1:
                        target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)

                    return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

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
                
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])
        return actions.FunctionCall(_NO_OP, [])

    def moveNumberTwo(self, obs, unit_type):
        smart_action, x, y = self.splitAction(self.previous_action)
            
        if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if _HARVEST_GATHER in obs.observation['available_actions']:
                unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
                
                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    
                    m_x = unit_x[i]
                    m_y = unit_y[i]
                    
                    target = [int(m_x), int(m_y)]
                    
                    return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
        return actions.FunctionCall(_NO_OP, [])

    def getCurrentState(self, obs, cc_count, supply_depot_count, barracks_count):
        current_state = np.zeros(8)
        current_state[0] = self.normalize(cc_count, 0, 1)
        current_state[1] = self.normalize(supply_depot_count, 0, 2)
        current_state[2] = self.normalize(barracks_count, 0, 2)
        army_supply = obs.observation['player'][_ARMY_SUPPLY]
        current_state[3] = self.normalize(army_supply, 0, 19)

        hot_squares = np.zeros(4)        
        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 32))
            x = int(math.ceil((enemy_x[i] + 1) / 32))
            
            hot_squares[((y - 1) * 2) + (x - 1)] += 1
        
        for i in range(len(hot_squares)):
            hot_squares[i] = self.normalize(hot_squares[i], 0, 30)

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]
        
        for i in range(0, 4):
            current_state[i + 4] = hot_squares[i]
        return current_state

    def normalize(self, value, min, max):
        value_loc = value
        min_loc = min
        max_loc = max

        if(value_loc > max_loc):
            value_loc = max_loc
        if(value_loc < min_loc):
            value_loc = min_loc

        value_loc += (min * -1)
        min_loc += (min * -1)
        max_loc += (min * -1)

        return ((value_loc / (max_loc-min_loc)) * 2) - 1

    def calculate_reward(self, state_object, last_killed_units, killed_units, last_killed_buildings, killed_buildings):
        reward = 0
        if(killed_units > last_killed_units):
            reward += KILL_UNIT_REWARD

        if(killed_buildings > last_killed_buildings):
            reward += KILL_BUILDING_REWARD

        if(state_object[0][3] < 0.4 and state_object[1] != 3):
            reward -= BUILD_FIGHTING_UNIT_REWARD

        if(reward < -1):
            reward = -1
        if(reward > 1):
            reward = 1
        state_object[2] = reward

        return state_object




        
