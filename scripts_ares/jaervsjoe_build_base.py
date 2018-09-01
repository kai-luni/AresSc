from dto.action_base_dto import ActionBaseDto
from helper_functions.obs_helper import get_current_state, get_random_unit, get_count_unit
from helper_functions.geo_helper import transform_distance

from pysc2.lib import actions, features, units

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

class JaervsjoeBuildBase:
    def __init__(self):
        #keep track of what action we are performin right now
        self.previous_action = None       

    def act_build_base(self, current_state_others):
        #TODO: build command center
        command_center_count = current_state_others[0]
        supply_depot_count = current_state_others[1]
        barracks_count = current_state_others[2]
        army_supply_count = current_state_others[3]
        # resources = current_state[4]

        # cost_command_center = 100
        # cost_supply_depot = 100
        # cost_barracks = 100

        if barracks_count < 0.9:
            return ActionBaseDto.build_barracks()
        if supply_depot_count < 0.9:
            return ActionBaseDto.build_supply_depot()
        if army_supply_count < 1:
            return ActionBaseDto.build_marine()

        return ActionBaseDto.do_nothing()

    def moveNumberZero(self, obs):      
        current_state = get_current_state(obs)
        if obs.last():
            raise Exception('last frame, not defined')

        smart_action = self.act_build_base(current_state["state_others"])
        self.previous_action = smart_action
        #activity = LastActivityAresDto(rl_action, 0, obs, current_state)

        
        if smart_action == ActionBaseDto.build_barracks() or smart_action == ActionBaseDto.build_supply_depot():
            scv = get_random_unit(obs, units.Terran.SCV)

            if(scv is None):
                raise Exception("could not select unit.")
            if(scv is not None):
                return actions.FUNCTIONS.select_point("select", (scv.x, scv.y))
            
        elif smart_action == ActionBaseDto.build_marine():
            barrack = get_random_unit(obs, units.Terran.Barracks)
            if(barrack is not None):
                return actions.FUNCTIONS.select_point("select_all_type", (barrack.x, barrack.y))

        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])


    def moveNumberOne(self, obs, base_upper_left):
        smart_action = self.previous_action
        commCenter = get_random_unit(obs, units.Terran.CommandCenter)

        if smart_action == ActionBaseDto.build_supply_depot():
            if (actions.FUNCTIONS.Build_SupplyDepot_screen.id in obs.observation.available_actions):
                if commCenter is not None:
                    supply_depot_count = get_count_unit(obs, units.Terran.SupplyDepot)
                    if supply_depot_count == 0:
                        target = transform_distance(round(commCenter.x), -35, round(commCenter.y), 0, base_upper_left)
                        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)
                    elif supply_depot_count == 1:
                        target = transform_distance(round(commCenter.x), -25, round(commCenter.y), -25, base_upper_left)
                        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)

        elif smart_action == ActionBaseDto.build_barracks():
            barracks_count = get_count_unit(obs, units.Terran.Barracks)
            if barracks_count < 2 and actions.FUNCTIONS.Build_Barracks_screen.id in obs.observation.available_actions:
                if commCenter is not None:
                    if  barracks_count == 0:
                        target = transform_distance(round(commCenter.x), 15, round(commCenter.y), -9, base_upper_left)
                    elif  barracks_count == 1:
                        target = transform_distance(round(commCenter.x), 15, round(commCenter.y), 12, base_upper_left)

                    return actions.FUNCTIONS.Build_Barracks_screen("now", target)

        elif smart_action == ActionBaseDto.build_marine():
            if _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
    
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    def moveNumberTwo(self, obs):
        smart_action = self.previous_action
            
        if smart_action == ActionBaseDto.build_barracks() or smart_action == ActionBaseDto.build_supply_depot():
            if (actions.FUNCTIONS.Harvest_Gather_screen.id in obs.observation.available_actions):
                mineral_field = get_random_unit(obs, units.Neutral.MineralField)
                if mineral_field is not None:
                    return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, [mineral_field.x, mineral_field.y]])

        return actions.FunctionCall(_NO_OP, [])

