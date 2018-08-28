from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from reward.reward_calculator import RewardCalculator
from helper_functions.normalizer import normalize
from helper_functions.obs_helper import get_random_unit, get_count_unit, get_excluded_actions

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

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'


class AresAgentNew(base_agent.BaseAgent):
    def __init__(self):
        last_activity = None


    def step(self, obs):
        super(AresAgentNew, self).step(obs)

        excluded_actions = get_excluded_actions(obs)

        if self.last_activity.move_number == 0:
            self.last_activity.move_number += 1
            value = self.moveNumberZero(obs, self.last_activity, excluded_actions)
        elif self.last_activity.move_number == 1:
            self.last_activity.move_number += 1
            value =  self.moveNumberOne(obs)
        elif self.last_activity.move_number == 2:
            self.last_activity.move_number = 0
            value =  self.moveNumberTwo(obs, unit_type)
        return value


    def moveNumberZero(self, obs, last_activity, excluded_actions):      
        current_state = self.getCurrentState(obs)

        if last_activity.action is None:
            raise Exception('previous_action is None')
        if obs.last():
            raise Exception('last frame, not defined')
        
        if self.previous_action is not None and not obs.last():
            killed_unit_score = obs.observation['score_cumulative'][5]
            killed_building_score = obs.observation['score_cumulative'][6]
            self.last_score = obs.observation['score_cumulative'][0]
            reward = self.reward_calc.get_reward_from_observation(last_activity.obs, last_activity.action)
            state_object = [last_activity.state, last_activity.action, reward, current_state, obs.last(), excluded_actions]
            self.last_killed_unit_score = killed_unit_score
            self.Last_killed_building_score = killed_building_score
            self.qlearn.memory_episode.append(state_object)

        rl_action = 0
        activity = LastActivityAresDto(rl_action, 0, obs, current_state)

        smart_action, _, _ = self.splitAction(self.previous_action)
        
        if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            scv = get_random_unit(obs, units.Terran.SCV)

            if(scv is None):
                raise Exception("could not select unit.")
            if(scv is not None):
                return actions.FUNCTIONS.select_point("select", (scv.x, scv.y))
            
        elif smart_action == ACTION_BUILD_MARINE:
            barrack = get_random_unit(obs, units.Terran.Barracks)
            if(barrack is not None):
                return actions.FUNCTIONS.select_point("select_all_type", (barrack.x, barrack.y))
            
        elif smart_action == ACTION_ATTACK:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        return actions.FunctionCall(_NO_OP, []), activity


    def getCurrentState(self, obs):
        """
        get an array with information about
        0: command center count
        1: supply depot count
        2: barracks count
        3: supply depot count
        normalized to be between -1 and 1
        """
        supply_depot_count = get_count_unit(obs, units.Terran.SupplyDepot)
        barracks_count = get_count_unit(obs, units.Terran.Barracks)
        current_state = []
        current_state.append(normalize(get_count_unit(obs, units.Terran.CommandCenter), 0, 1))
        current_state.append(normalize(supply_depot_count, 0, 2))
        current_state.append(normalize(barracks_count, 0, 2))
        army_supply = obs.observation['player'][_ARMY_SUPPLY]
        current_state.append(normalize(army_supply, 0, 19))
        test_pure = (obs.observation['rgb_minimap'] / 128) - 1

        return_dict = {}
        return_dict["state_enemy_matrix"] = test_pure
        return_dict["state_others"] = np.array(current_state)
        return return_dict

class LastActivityAresDto:
    def __init__(self, action, move_number, obs, state):
        self.obs = obs
        self.action = action
        self.move_number = move_number
        self.state = state



