@staticmethod
def transform_distance(x, x_distance, y, y_distance, base_top_left):
    """helps to get correct coords for building, depending on the location of the base"""
    if not base_top_left:
        return [x - x_distance, y - y_distance]
    
    return [x + x_distance, y + y_distance]