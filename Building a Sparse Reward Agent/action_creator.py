
from point_rect import Rect, Point
from map_matrix import get_eight_by_eight_matrix

class action_creator:
    def get_attack_actions(map_matrix):
        return_array = []
        map_matrix = get_eight_by_eight_matrix(64, 64)
        for height in range(8):
            for width in range(8):
                return_array.append(ACTION_ATTACK + '_' + str(map_matrix[height][width].get_center().x) + '_' + str(map_matrix[height][width].get_center().y))

        return return_array