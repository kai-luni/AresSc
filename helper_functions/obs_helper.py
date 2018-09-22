import random

import numpy as np

from helper_functions.normalizer import normalize
from pysc2.lib import actions, features, units

def get_current_state(obs):
    """
    get an array with information about
    0: command center count
    1: supply depot count
    2: barracks count
    3: army supply count
    normalized to be between -1 and 1
    """
    _ARMY_SUPPLY = 5

    supply_depot_count = get_count_unit(obs, units.Terran.SupplyDepot)
    barracks_count = get_count_unit(obs, units.Terran.Barracks)
    current_state = []
    current_state.append(normalize(get_count_unit(obs, units.Terran.CommandCenter), 0, 1))
    current_state.append(normalize(supply_depot_count, 0, 2))
    current_state.append(normalize(barracks_count, 0, 2))
    army_supply = obs.observation['player'][_ARMY_SUPPLY]
    current_state.append(normalize(army_supply, 0, 19))
    rgb_minimap = (obs.observation['rgb_minimap'] / 128) - 1

    #raise Exception("4: resources")

    return_dict = {}
    return_dict["state_enemy_matrix"] = rgb_minimap
    return_dict["state_others"] = np.array(current_state)
    return return_dict

def get_excluded_actions(obs):
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
    
    return excluded_actions

def get_random_unit(obs, unit_id):
    """
    get a random unit of a certain kind, buildings included
    obs: pysc2 observation
    unit_id(int): id of unit
    returns: None or unit (object?)
    """
    units = [unit for unit in obs.observation.feature_units if unit.unit_type == unit_id]
    if len(units) > 0:
        return random.choice(units)
    return None

def get_count_unit(obs, unit_id):
    """
    get count of a unit of a certain kind, buildings included
    obs: pysc2 observation
    unit_id(int): id of unit
    returns: (int) count
    """
    units = [unit for unit in obs.observation.feature_units if unit.unit_type == unit_id]
    return len(units)

def base_is_upper_left(obs):
    """
    get position of base
    return: (bool) true: upper left, fasle: lower right
    """
    player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
    xmean = player_x.mean()
    ymean = player_y.mean()
    if xmean <= 31 and ymean <= 31:
        return True
    return False

