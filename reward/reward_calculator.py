from reward.kdh_state_dto import KdhStateDto
from helper_functions.normalizer import normalize
from helper_functions.obs_helper  import get_count_unit


from pysc2.lib import actions, features, units

KILL_UNIT_REWARD = 0.5
KILL_BUILDING_REWARD = 0.9
BUILD_FIGHTING_UNIT_REWARD = 0.6
BUILD_IMPORTANT_BUILDING_REWARD = 0.8

LOOSE_BUILDING_PENALTY = -0.9
LOOSE_FIGHTING_UNIT_PENALTY = -0.5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_ARMY_SUPPLY = 5



class RewardCalculator:
    def __init__(self):
        self.last_kdh_state = None

    def get_reward(self, kdh_state):

        if(self.last_kdh_state == None):
            self.last_kdh_state = kdh_state
            return 0
        reward = 0
        if(kdh_state.killed_enemies > self.last_kdh_state.killed_enemies):
            reward += KILL_UNIT_REWARD

        if(kdh_state.destroyed_buildings > self.last_kdh_state.destroyed_buildings):
            reward += KILL_BUILDING_REWARD

        if(kdh_state.own_army < 0.4 and kdh_state.action == 3):
            reward += BUILD_FIGHTING_UNIT_REWARD

        if(kdh_state.own_buildings < self.last_kdh_state.own_buildings):
            reward += LOOSE_BUILDING_PENALTY

        if(kdh_state.own_barracks < 2 and kdh_state.action == 2):
            reward += BUILD_IMPORTANT_BUILDING_REWARD

        if(kdh_state.own_depot < 2 and kdh_state.action == 1):
            reward += BUILD_IMPORTANT_BUILDING_REWARD

        if(reward < -1):
            reward = -1
        if(reward > 1):
            reward = 1

        self.last_kdh_state = kdh_state

        return reward

    def get_reward_from_observation(self, obs, action):
        state_dto = KdhStateDto()

        state_dto.action = action
        state_dto.killed_enemies = obs.observation['score_cumulative'][5]
        state_dto.destroyed_buildings = obs.observation['score_cumulative'][6]


        unit_type = obs.observation['rgb_screen'][_UNIT_TYPE]
        # cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        # cc_count = 1 if cc_y.any() else 0
        cc_count = get_count_unit(obs, units.Terran.CommandCenter)
        # depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        # supply_depot_count = int(round(len(depot_y) / 69))
        supply_depot_count = get_count_unit(obs, units.Terran.SupplyDepot)
        # barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        # barracks_count = int(round(len(barracks_y) / 137))
        barracks_count = get_count_unit(obs, units.Terran.Barracks)

        state_dto.own_barracks = barracks_count
        state_dto.own_depot = supply_depot_count
        state_dto.own_buildings = normalize(cc_count + supply_depot_count + barracks_count, 0, 5)


        army_supply = obs.observation['player'][_ARMY_SUPPLY]
        state_dto.own_army = normalize(army_supply, 0, 19)

        return self.get_reward(state_dto)