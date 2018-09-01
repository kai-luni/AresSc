"""Run some basic tests, minimize bugs"""

from map_matrix import get_eight_by_eight_matrix
from neural_networks.ares_net_attack import AresNetAttack
from unit_test.test_brain import TestBrain

def test_map_matrix():
    """test the module map_matrix"""
    matrix = get_eight_by_eight_matrix(64, 64)


    if(matrix[0][0].overlaps(matrix[0][1])):
        raise Exception("map matrix fields should not overlap")

    if(matrix[1][1].overlaps(matrix[2][1])):
        raise Exception("map matrix fields should not overlap")
    
    if(len(matrix[0]) != 8 or len(matrix) != 8):
        raise Exception("the size of this matrix need to by 8 by 8")
        
    if(matrix[1][1].top != 8 or matrix[1][1].bottom != 15 or matrix[1][1].left != 8 or matrix[1][1].right != 15):
        raise Exception("the matrix field one one is wrong")

def test__ddqn_data_creation():
    batch_list = []
    for i in range(10):
        batch_item = []
        batch_item.append({"state_enemy_matrix": i, "state_others": i})
        batch_item.append(i)#action
        batch_item.append(i)
        batch_item.append({"state_enemy_matrix": i*2, "state_others": i*2})
        if i is 9:
            batch_item.append(True)
        else:
            batch_item.append(False)
        batch_list.append(batch_item)

    loss, brain = AresNetAttack.create_ddqn_data(batch_list, TestBrain("brain"), TestBrain("target"), 0.5)

    x_brain = brain.get_x()
    y_brain = brain.get_y()
    for i in range(len(y_brain)):
        print("entry " + str(i))
        print("x other: " + str(x_brain[0][i]) + " x pic: " + str(x_brain[1][i]) + " y " + str(y_brain[i]))

    



#test_map_matrix()
test__ddqn_data_creation()





