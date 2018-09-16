import random
import gym
import numpy as np

from gym import spaces
from rl.core import Env
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

from scripts_ares.jaervsjoe_build_base import JaervsjoeBuildBase
from map_matrix import get_eight_by_eight_matrix, get_coordinates_by_index
from reward.reward_calculator import RewardCalculator
from memory_episode_helper import MemoryEpisodeHelper
from ares_processor import AresProcessor


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

class AresEnvGym(gym.Env):
    def __init__(self, input_shape):
        super(AresEnvGym, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(65)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=input_shape, dtype=np.uint8)

        self.attack = False
        self.move_number = 0
        self.map_matrix = get_eight_by_eight_matrix(64, 64)
        self.reward_calculator = RewardCalculator()

        self.build_Bot = JaervsjoeBuildBase()


        self.ares_processor = AresProcessor(input_shape)

        #this is the pysc2 environment that interacts with the game
        self.pysc2_env = sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],
                agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=84, minimap=64), rgb_dimensions=features.Dimensions(screen=384, minimap=64), action_space=actions.ActionSpace.FEATURES, use_feature_units=True),
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
                self.last_obs = self.pysc2_env.reset()[0]
                return self.last_obs, obs.reward, True, {}

            if obs.first():
                #important: reset reward calculator
                self.reward_calculator = RewardCalculator()
            reward += self.reward_calculator.get_reward_from_observation(obs)

            if i == 0:
                value = self.build_Bot.moveNumberZero(obs)
            elif i == 1:
                value =  self.build_Bot.moveNumberOne(obs)
            elif i == 2:
                value =  self.build_Bot.moveNumberTwo(obs)
            elif action == 64:
                #action 64 is no action
                value =  actions.FunctionCall(_NO_OP, [])
            elif i == 3:
                value = self.moveNumberZero(obs)
            elif i == 4:
                value =  self.moveNumberOne(obs, action)
            elif i == 5:
                value =  actions.FunctionCall(_NO_OP, [])
            obs = self.pysc2_env.step([value])[0]

        self.last_obs = obs

        picture = self.ares_processor.process_observation(obs)
        return picture, reward, False, {}

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        return self.pysc2_env.reset()[0]


    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        print('do nothing')

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self.pysc2_env.close()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

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