
from point_rect import Point, Rect

class MapMatrix:
    def get_eight_by_eight_matrix(self, map_width, map_height):
        return_matrix = []
        width_sub_rect = map_width / 8
        height_sub_rect = map_height / 8
        start_x = 0
        start_y = 0
        for index_height in range(8):
            width_array = []
            for index_width in range(8):
                width_array.append(Rect(Point(start_x, start_y), Point(start_x+width_sub_rect, start_y+height_sub_rect)))
                start_x += width_sub_rect
            return_matrix.append(width_array)
            width_array = []
            start_x = 0
            start_y += height_sub_rect

