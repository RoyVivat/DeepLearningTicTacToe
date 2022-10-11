import numpy as np
from typing import List
from game import Player, TurnBasedGame
from mctsplayer import MCTSPlayer

class TicTacToe(TurnBasedGame):
    def __init__(self, game_state = None):
        super().__init__(game_state)
    
    def get_start_board(self):
        return np.zeros([3,3])
    
    def init_players(self, players):
        self.game_state['players'] = players
        self.game_state['players'][0].id = 1
        self.game_state['players'][1].id = 2

    def is_valid_move(self, move):
        # Checks that the move is a list or tuple of length 2 and that the placement is on an empty tile.
        if type(move) is not list and type(move) is not tuple:
            return False

        if not len(move) == 2:
            return False
         
        if not self.game_state['board'][move] == 0:
            return False
        
        return True
    
    def get_next_board(self, board, move):
        board = board.copy()
        board[move] = self.get_player().id
        return board
    
    def is_game_over(self):
        board = self.game_state['board']
        won = False
        winner = None

        for i in range(3):
            if np.all(board[i] == board[i][0]) and board[i][0] != 0:
                won = True
                result = 1 - (board[i][0] - 1)
                break
            if np.all(board[:, i] == board[0][i]) and board[0][i] != 0:
                won = True
                result = 1 - (board[0][i] - 1)
                break

        if board[0,0] == board[1,1] == board[2,2] and board[0][0] != 0:
            won = True
            result = 1 - (board[0][0] - 1)
        elif board[0,2] == board[1,1] == board[2,0] and board[0][2] != 0:
            won = True
            result = 1 - (board[0][2] - 1)

        if won:
            self.game_state['result'] = result
            return True
        
        if not (board == 0).any():
            self.game_state['result'] = 0.5
            return True

        return False 

    def render(self):
        print(self.game_state['board'])
        print()

    @staticmethod
    def get_valid_moves(board):
        
        return [tuple(i) for i in np.argwhere(board==0)]


# Picks all moves randomly
class RandomTTTPlayer(Player):
    def __init__(self, name, game):
            self.name = name
            self.game = game
            
    def play(self, game_state):
        valid_moves = self.game().get_valid_moves(game_state['board'])
        return valid_moves[np.random.randint(0, len(valid_moves))]

# Tries to grab the center as quickly as possible and then plays randomly
class SimpleTTTPlayer(Player):
    def __init__(self, name):
        self.name = name
    
    def play(self, gamestate):
        valid_moves = TicTacToe().get_valid_moves(gamestate)
        valid_moves = [list(coord) for coord in valid_moves]

        if [1, 1] in valid_moves:
            return (1, 1)

        return tuple(valid_moves[np.random.randint(0, len(valid_moves))]) 

class UserTTTPlayer(Player):
    def __init__(self, name, game):
        self.name = name
    
    def play(self, game_state):
        inp = input()
        return (int(inp[0]), int(inp[2]))

def main():
    results = 0

    for i in range(1):

        #game_state = {'board':np.array([[0,1,2],[0,0,0],[1,1,2]]), 'running':True, 'curr_player':1}
        
        T = TicTacToe()

        p1 = MCTSPlayer("p1", TicTacToe)
        p2 = UserTTTPlayer("p2", TicTacToe)

        T.init_players([p1, p2])
        T.run(render=True)
        #T.render()

        results += T.game_state['result']
        #print(T.game_state['result'])

    print(results)

if __name__ == '__main__':
    main()