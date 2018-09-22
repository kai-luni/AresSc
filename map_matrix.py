from point_rect import Point, Rect


def get_eight_by_eight_matrix(map_width, map_height):
    return_matrix = []
    width_sub_rect = map_width / 8
    height_sub_rect = map_height / 8
    start_x = 0
    start_y = 0
    for index_height in range(8):
        width_array = []
        for index_width in range(8):
            width_array.append(Rect(Point(start_x, start_y), Point(start_x+width_sub_rect-1, start_y+height_sub_rect-1)))
            start_x += width_sub_rect
        return_matrix.append(width_array)
        width_array = []
        start_x = 0
        start_y += height_sub_rect
    return return_matrix


def get_coordinates_by_index(matrix, index):
    """ Get proper coordinates for requested index

    #arguments
        matrix: array created with get_eight_by_eight_matrix
        index int from 0 to 63

    #returns
        coordinates on the map for appropriate area
    """

    #TODO: vary coordiantes a bit
    if(index > 63):
        raise Exception('out of bounds: ' + str(index))
    i = 0
    for height in range(8):
        for width in range(8):
            if(i == index):
                return matrix[height][width].get_random_point()
            i += 1
    raise Exception('out of bounds: ' + str(index))

