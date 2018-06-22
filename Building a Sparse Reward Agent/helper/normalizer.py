def normalize(value, min_val, max_val):
    """
    normalize value to -1...1
    example: 
    min: 2, max: 10, value: 10, return: 1
    min: 2, max: 10, value: 2, return: -1
    """
    value_loc = value
    min_loc = min_val
    max_loc = max_val

    if(value_loc > max_loc):
        value_loc = max_loc
    if(value_loc < min_loc):
        value_loc = min_loc

    value_loc += (min * -1)
    min_loc += (min * -1)
    max_loc += (min * -1)

    return ((value_loc / (max_loc-min_loc)) * 2) - 1