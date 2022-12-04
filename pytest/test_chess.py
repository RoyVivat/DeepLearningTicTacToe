import numpy as np

import sys
sys.path.append('.')

from games.chess import Chess

def test_valid_coodinate():
    c = Chess()

    # Test board extremes
    assert(c.is_valid_coordinate([0,0,0,7]) == True)
    assert(c.is_valid_coordinate([0,7,7,7]) == True)
    assert(c.is_valid_coordinate([7,7,7,0]) == True)
    assert(c.is_valid_coordinate([7,0,0,0]) == True)

    # Test duplicates
    assert(c.is_valid_coordinate([0,0,0,0]) == False)

    # Test random spots in the board
    assert(c.is_valid_coordinate([1,3,0,7]) == True)
    assert(c.is_valid_coordinate([1,6,2,7]) == True)
    assert(c.is_valid_coordinate([7,6,1,5]) == True)
    assert(c.is_valid_coordinate([2,3,2,5]) == True)
    assert(c.is_valid_coordinate([4,5,0,0]) == True)

    # Test outside of board
    assert(c.is_valid_coordinate([0,0,0,-1]) == False)
    assert(c.is_valid_coordinate([0,0,0,8]) == False)
    assert(c.is_valid_coordinate([-1,-2,-10,-100]) == False)
    assert(c.is_valid_coordinate([11,12,13,14]) == False)

def test_valid_bishop():
    c = Chess()

    c.board[4,4] = 'wb'

    # Test single move in any direction
    assert(c.is_valid_bishop_move([4,4,5,5]) == True)
    assert(c.is_valid_bishop_move([4,4,3,3]) == True)
    assert(c.is_valid_bishop_move([4,4,3,5]) == True)
    assert(c.is_valid_bishop_move([4,4,5,3]) == True)

    # Test all extremes
    assert(c.is_valid_bishop_move([4,4,7,7]) == False)
    assert(c.is_valid_bishop_move([4,4,0,0]) == False)
    assert(c.is_valid_bishop_move([4,4,1,7]) == True)
    assert(c.is_valid_bishop_move([4,4,7,1]) == False)

    # Test all edges
    assert(c.is_valid_bishop_move([4,4,3,4]) == False)
    assert(c.is_valid_bishop_move([4,4,4,3]) == False)
    assert(c.is_valid_bishop_move([4,4,4,5]) == False)
    assert(c.is_valid_bishop_move([4,4,5,4]) == False)

    # Test random
    assert(c.is_valid_bishop_move([4,4,5,6]) == False)
    assert(c.is_valid_bishop_move([4,4,0,7]) == False)
    assert(c.is_valid_bishop_move([4,4,3,2]) == False)

def test_valid_rook():
    pass