import numpy as np
from typing import List
from game import Player, TurnBasedGame

class TicTacToe(TurnBasedGame):
    def __init__(self):
        super().__init__()

    def init_gamestate(self):
        return np.zeros([3,3])
    
    def init_players(self, players: List[Player]):
        self.players = players
        self.players[0].id = 1
        self.players[1].id = -1

    def update_gamestate(self, move):
        if not self.is_valid_move(move):
            raise Exception("Invalid move:", move)
        
        self.gamestate[move] = self.curr_player.id

        if self.check_for_win():
            self.running = False
            self.winner = self.curr_player.name

        if self.get_valid_moves(self.gamestate).shape[0] == 0:
            self.running = False
            self.winner = "Tie"

    def check_for_win(self):
        for i in range(3):
            if (self.gamestate[i] == self.curr_player.id).all():
                return True
            if (self.gamestate[:, i] == self.curr_player.id).all():
                return True
        if self.gamestate[0,0] == self.gamestate[1,1] == self.gamestate[2,2] == self.curr_player.id:
            return True
        if self.gamestate[0,2] == self.gamestate[1,1] == self.gamestate[2,0] == self.curr_player.id:
            return True
        return False

    def is_valid_move(self, move) -> bool:
        if not len(move) == 2:
            return False
        
        if not self.gamestate[move] == 0:
            return False
        
        return True
    
    @staticmethod
    def get_valid_moves(gamestate):
        return np.argwhere(gamestate==0)
        
    def render(self):
        print(self.gamestate)
        print()


class SimpleTTTPlayer(Player):
    def __init__(self, name):
        self.name = name

    def play(self, gamestate):
        valid_moves = TicTacToe().get_valid_moves(gamestate)
        print(valid_moves.shape)
        return tuple(valid_moves[np.random.randint(0, valid_moves.shape[0])])

T = TicTacToe()
p1 = SimpleTTTPlayer("p1")
p2 = SimpleTTTPlayer("p2")

T.init_players([p1, p2])
T.run()
print(T.winner)