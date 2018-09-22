import random
import pickle
import os

import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from reward.reward_calculator import RewardCalculator
from helper_functions.normalizer import normalize
from helper_functions.obs_helper import get_random_unit, get_count_unit, get_excluded_actions, get_current_state
from scripts_ares.jaervsjoe_build_base import JaervsjoeBuildBase

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_ARMY_SUPPLY = 5

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'


class AresAgentNew(base_agent.BaseAgent):
    def __init__(self):
        self.last_activity = None
        #switch every 3 tripple step between not attack (build) and attack
        self.attack = False
        self.move_number = 0
        self.episodes = 0
        self.steps = 0
        self.reward = 0

        self.build_Bot = JaervsjoeBuildBase()

    def step(self, obs):
        super(AresAgentNew, self).step(obs)

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

        if not self.attack:
            if self.move_number == 0:
                self.move_number += 1
                value = self.build_Bot.moveNumberZero(obs)
            elif self.move_number == 1:
                self.move_number += 1
                value =  self.build_Bot.moveNumberOne(obs, self.base_top_left)
            elif self.move_number == 2:
                self.move_number = 0
                self.attack = not self.attack
                value =  self.build_Bot.moveNumberTwo(obs)
            else:
                value = actions.FunctionCall(_NO_OP, [])
        else:
            if self.move_number == 0:
                self.move_number += 1
                value = actions.FunctionCall(_NO_OP, [])
            elif self.move_number == 1:
                self.move_number += 1
                value =  actions.FunctionCall(_NO_OP, [])
            elif self.move_number == 2:
                self.move_number = 0
                self.attack = not self.attack
                value =  actions.FunctionCall(_NO_OP, [])
            else:
                value = actions.FunctionCall(_NO_OP, [])
        return value


    # def moveNumberZero(self, obs, last_activity, excluded_actions):      
    #     current_state = get_current_state(obs)

    #     if last_activity.action is None:
    #         raise Exception('previous_action is None')
    #     if obs.last():
    #         raise Exception('last frame, not defined')
        
    #     if self.previous_action is not None and not obs.last():
    #         killed_unit_score = obs.observation['score_cumulative'][5]
    #         killed_building_score = obs.observation['score_cumulative'][6]
    #         self.last_score = obs.observation['score_cumulative'][0]
    #         reward = self.reward_calc.get_reward_from_observation(last_activity.obs, last_activity.action)
    #         state_object = [last_activity.state, last_activity.action, reward, current_state, obs.last(), excluded_actions]
    #         self.last_killed_unit_score = killed_unit_score
    #         self.Last_killed_building_score = killed_building_score
    #         self.qlearn.memory_episode.append(state_object)

    #     rl_action = 0
    #     activity = LastActivityAresDto(rl_action, 0, obs, current_state)

    #     smart_action, _, _ = self.splitAction(self.previous_action)
        
    #     if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
    #         scv = get_random_unit(obs, units.Terran.SCV)

    #         if(scv is None):
    #             raise Exception("could not select unit.")
    #         if(scv is not None):
    #             return actions.FUNCTIONS.select_point("select", (scv.x, scv.y))
            
    #     elif smart_action == ACTION_BUILD_MARINE:
    #         barrack = get_random_unit(obs, units.Terran.Barracks)
    #         if(barrack is not None):
    #             return actions.FUNCTIONS.select_point("select_all_type", (barrack.x, barrack.y))
            
    #     elif smart_action == ACTION_ATTACK:
    #         if _SELECT_ARMY in obs.observation['available_actions']:
    #             return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

    #     return actions.FunctionCall(_NO_OP, []), activity

    # def moveNumberOne(self, obs):
    #     smart_action, x, y = self.splitAction(self.previous_action)
    #     commCenter = get_random_unit(obs, units.Terran.CommandCenter)

    #     if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
    #         if (actions.FUNCTIONS.Build_SupplyDepot_screen.id in obs.observation.available_actions):
    #             if commCenter is not None:
    #                 supply_depot_count = get_count_unit(obs, units.Terran.SupplyDepot)
    #                 if supply_depot_count == 0:
    #                     target = self.transformDistance(round(commCenter.x), -35, round(commCenter.y), 0)
    #                     return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)
    #                 elif supply_depot_count == 1:
    #                     target = self.transformDistance(round(commCenter.x), -25, round(commCenter.y), -25)
    #                     return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)

    #     elif smart_action == ACTION_BUILD_BARRACKS:
    #         barracks_count = get_count_unit(obs, units.Terran.Barracks)
    #         if barracks_count < 2 and actions.FUNCTIONS.Build_Barracks_screen.id in obs.observation.available_actions:
    #             if commCenter is not None:
    #                 if  barracks_count == 0:
    #                     target = self.transformDistance(round(commCenter.x), 15, round(commCenter.y), -9)
    #                 elif  barracks_count == 1:
    #                     target = self.transformDistance(round(commCenter.x), 15, round(commCenter.y), 12)

    #                 return actions.FUNCTIONS.Build_Barracks_screen("now", target)

    #     elif smart_action == ACTION_BUILD_MARINE:
    #         if _TRAIN_MARINE in obs.observation['available_actions']:
    #             return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
    
    #     elif smart_action == ACTION_ATTACK:
    #         do_it = True
            
    #         if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == units.Terran.SCV:
    #             do_it = False
            
    #         if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == units.Terran.SCV:
    #             do_it = False
            
    #         if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
    #             x_offset = random.randint(-1, 1)
    #             y_offset = random.randint(-1, 1)
                
    #             return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(float(x) + (x_offset * 2), float(y) + (y_offset * 2))])
    #     return actions.FunctionCall(_NO_OP, [])


    # def moveNumberTwo(self, obs):
    #     smart_action, _, _ = self.splitAction(self.previous_action)
            
    #     if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
    #         if (actions.FUNCTIONS.Harvest_Gather_screen.id in obs.observation.available_actions):
    #             mineral_field = get_random_unit(obs, units.Neutral.MineralField)
    #             if mineral_field is not None:
    #                 return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, [mineral_field.x, mineral_field.y]])

    #     return actions.FunctionCall(_NO_OP, [])

class LastActivityAresDto:
    def __init__(self, action, move_number, obs, state):
        self.obs = obs
        self.action = action
        self.move_number = move_number
        self.state = state







