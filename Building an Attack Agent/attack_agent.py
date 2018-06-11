import random
import math

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from doubleDQNAgent import DoubleDQNAgent
from networks import Networks
from qAgent import qAgent

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_QUEUED = [1]

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5

class AttackAgent(base_agent.BaseAgent):
    def __init__(self):
        super(AttackAgent, self).__init__()

        self.max_memory = 50000
        
        action_size = len(smart_actions)
        trace_length = 32
        state_size = 20
        
        #self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.qlearn = qAgent(state_size = state_size, action_size = action_size)
        #self.qlearn.model = Networks.drqn(state_size, action_size, self.qlearn.learning_rate)
        #self.qlearn.target_model = Networks.drqn(state_size, action_size, self.qlearn.learning_rate)
        
        self.episode_buf = [] # Save entire episode

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        
        self.previous_action = None
        self.previous_state = None


    def transformDistance(self, x, x_distance, y, y_distance):
        returnValue = None
        if not self.base_top_left:
            returnValue = [x - x_distance, y - y_distance]
        else:
            returnValue = [x + x_distance, y + y_distance]
        if(returnValue[0] < 0):
            returnValue[0] = 0
        if(returnValue[1] < 0):
            returnValue[1] = 0
        return returnValue
    
    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]
        
        return [x, y]
        
    def step(self, obs):
        super(AttackAgent, self).step(obs)

        # save progress every 10000 iterations
        # if self.steps % 10000 == 0:
        #     print("Now we save model")
        #     self.qlearn.model.save_weights("models/drqn.h5", overwrite=True)

        
        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
        unit_type = obs.observation['screen'][_UNIT_TYPE]

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = supply_depot_count = 1 if depot_y.any() else 0

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = 1 if barracks_y.any() else 0
            
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        
        current_state = np.zeros(20)
        current_state[0] = supply_depot_count
        current_state[1] = barracks_count
        current_state[2] = supply_limit
        current_state[3] = army_supply

        hot_squares = np.zeros(16)        
        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))
            
            hot_squares[((y - 1) * 4) + (x - 1)] = 1
        
        if not self.base_top_left:
            hot_squares = hot_squares[::-1]
        
        for i in range(0, 16):
            current_state[i + 4] = hot_squares[i]

        stateObject = None
        if self.previous_action is not None:
            reward = 0
                
            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD
                    
            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD
                
            # save the sample <s, a, r, s', done> to episode buffer
            #s: observation
            #r: reward
            #s': new state
            #done (can out as this is by episodes (?))
            stateObject = [self.previous_state, self.previous_action, reward, current_state, obs.last()]
            self.qlearn.memory.append(stateObject)

            #old
            #self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
        


        self.qlearn.replay(32)
        # # Do the training
        # if self.episodes > self.qlearn.batch_size:
        #     # Update epsilon
        #     if self.qlearn.epsilon > self.qlearn.final_epsilon and self.steps > self.qlearn.observe:
        #         self.qlearn.epsilon -= (self.qlearn.initial_epsilon - self.qlearn.final_epsilon) / self.qlearn.explore            
        #     Q_max, loss = self.qlearn.train_replay()
        


        if obs.last():
            #self.qlearn.memory.add(self.episode_buf)
            self.episode_buf = [] # Reset Episode Buf

        # if len(self.episode_buf) > self.qlearn.trace_length:
        #     # 1x8x64x64x3
        #     state_series = np.array([trace[-1] for trace in self.episode_buf[-self.qlearn.trace_length:]])
        #     state_series = np.expand_dims(state_series, axis=0)
        #     rl_action  = self.qlearn.get_action(state_series)
        # else:
        #     rl_action = random.randrange(self.qlearn.action_size)
        #rl_action = self.qlearn.choose_action(str(current_state))
        rl_action = self.qlearn.act(current_state)
        smart_action = smart_actions[rl_action]
        
        if(self.steps%100 == 0):
            print("Epsilon " + str(self.qlearn.exploration_rate) + " Action " + smart_action)

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action
        
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')
            
        if smart_action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])

        elif smart_action == ACTION_SELECT_SCV:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        
        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                
                if unit_y.any():
                    target = self.transformDistance(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                
                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
        
        elif smart_action == ACTION_BUILD_BARRACKS:
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                
                if unit_y.any():
                    target = self.transformDistance(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
            
                    return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
    
        elif smart_action == ACTION_SELECT_BARRACKS:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                
            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]
        
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        
        elif smart_action == ACTION_BUILD_MARINE:
            if _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        
        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        
        elif smart_action == ACTION_ATTACK:
            if obs.observation['single_select'][0][0] != _TERRAN_SCV and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x), int(y))])
        
        return actions.FunctionCall(_NO_OP, [])
