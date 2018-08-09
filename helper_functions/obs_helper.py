import random

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
