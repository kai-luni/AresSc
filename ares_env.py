import random

from rl.core import Env
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

from scripts_ares.jaervsjoe_build_base import JaervsjoeBuildBase
from map_matrix import get_eight_by_eight_matrix, get_coordinates_by_index
from reward.reward_calculator import RewardCalculator


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

class AresEnv(Env):
    def __init__(self):
        self.attack = False
        self.move_number = 0
        self.map_matrix = get_eight_by_eight_matrix(64, 64)
        self.reward_calculator = RewardCalculator()

        self.build_Bot = JaervsjoeBuildBase()

        #this is the pysc2 environment that interacts with the game
        self.pysc2_env = sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],
                agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=84, minimap=64), rgb_dimensions=features.Dimensions(screen=196, minimap=64), action_space=actions.ActionSpace.FEATURES, use_feature_units=True),
                step_mul=8,
                game_steps_per_episode=0,
                visualize=False)   
        self.last_obs = self.pysc2_env.reset()[0]     
    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs = self.last_obs
        reward = 0.0
        #each roundtrip consists of 6 steps, 3 attack and 3 build steps
        for i in range(6):
            
            if obs.last():
                self.last_obs = self.pysc2_env.reset()
                return self.last_obs, obs.reward, True, None

            if obs.first():
                #TODO: change to pysc2 v2
                player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
                xmean = player_x.mean()
                ymean = player_y.mean()        
                if xmean <= 31 and ymean <= 31:
                    self.base_top_left = 1
                else:
                    self.base_top_left = 0

                #important: reset reward calculator
                self.reward_calculator = RewardCalculator()

            reward += self.reward_calculator.get_reward_from_observation(obs)

            if i == 0:
                value = self.build_Bot.moveNumberZero(obs)
            elif i == 1:
                value =  self.build_Bot.moveNumberOne(obs, self.base_top_left)
            elif i == 2:
                value =  self.build_Bot.moveNumberTwo(obs)
            if i == 3:
                value = self.moveNumberZero(obs)
            elif i == 4:
                value =  self.moveNumberOne(obs, action)
            elif i == 5:
                value =  actions.FunctionCall(_NO_OP, [])
            obs = self.pysc2_env.step(value)

        self.last_obs = obs
        return obs, reward, False, None

    def moveNumberZero(self, obs):
        """select all fighting units"""
        if _SELECT_ARMY in obs.observation['available_actions']:
            return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        return actions.FunctionCall(_NO_OP, [])

    def moveNumberOne(self, obs, rl_action):
        """attack from neural network chosen location"""
        attack_point = get_coordinates_by_index(self.map_matrix, rl_action)

        do_it = True
        
        if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == units.Terran.SCV:
            do_it = False
        
        if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == units.Terran.SCV:
            do_it = False
        
        if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
            return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [float(attack_point.x), float(attack_point.y)]])
        return actions.FunctionCall(_NO_OP, [])