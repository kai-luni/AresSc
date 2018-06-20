from map_matrix import MapMatrix

def test_map_matrix():
    """test the module map_matrix"""
    map_matrix = MapMatrix()
    matrix = map_matrix.get_eight_by_eight_matrix(64, 64)

    if(matrix[0][0].overlaps(matrix[0][1])):
        raise(Exception("map matrix fields should not overlap"))

    if(matrix[1][1].overlaps(matrix[2][1])):
        raise(Exception("map matrix fields should not overlap"))
    
    if(len(matrix[0]) != 8 or len(matrix) != 8):
        raise(Exception("the size of this matrix need to by 8 by 8"))
        
    if(matrix[1][1].top != 8 or matrix[1][1].bottom != 15 or matrix[1][1].left != 8 or matrix[1][1].right != 15):
        raise(Exception("the matrix field one one is wrong"))

test_map_matrix()