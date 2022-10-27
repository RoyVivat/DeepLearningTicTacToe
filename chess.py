import numpy as np
from typing import List
from game import Player, TurnBasedGame
from mctsplayer import MCTSPlayer

class Chess(TurnBasedGame):
    def __init__(self, game_state = None):
        super().__init__(game_state)
    
    @staticmethod
    def get_start_board():
        return np.array(
            [['br', 'bn', 'bb', 'bq', 'bk', 'bb', 'bn', 'br'],
             ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
             ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
             ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
             ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
             ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
             ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
             ['wr', 'wn', 'wb', 'wq', 'wk', 'wb', 'wn', 'wr']]
        )

    def is_game_over(self):
        board = self.board

        if self.is_in_check():
            pass

c = Chess()