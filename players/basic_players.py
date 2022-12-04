import numpy as np

import sys
sys.path.append('.')

from games.game import Player, TurnBasedGame

# Picks all moves randomly
class RandomPlayer(Player):
    def __init__(self, game: TurnBasedGame):
            self.game = game
            
    def play(self, game_state):
        g = self.game(game_state)
        valid_moves = g.get_valid_moves()
        return valid_moves[np.random.randint(0, len(valid_moves))]

# Allows users to pick moves for tictactoe
class UserTTTPlayer(Player):
    def play(self, game_state):
        inp = input()
        return (int(inp[0]), int(inp[2]))