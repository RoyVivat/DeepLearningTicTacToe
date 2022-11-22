#import tensorflow as tf
import numpy as np

import sys
sys.path.append('.')
import time
import absl.logging
import os
import glob
from enum import IntEnum

from games.game import Player, TurnBasedGame, Turn, Result
from players.mctsplayer import MCTSPlayer
from players.alphazero import AlphaZeroPlayer

#absl.logging.set_verbosity(absl.logging.ERROR)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class TicTacToe(TurnBasedGame):
    def __init__(self, game_state = None):
        super().__init__(game_state)
        self.players = {Turn.P1:None, Turn.P2:None}
    
    @staticmethod
    def get_start_board():
        # Returns the initial game board
        return np.zeros([3,3])
    
    def init_players(self, players):
        self.players = {1: players[0], -1: players[1]}

    def is_game_over(self):
        # Checks if the game is in a terminating state
        board = self.board
        won = False
        result = None

        # Check rows and columns
        for i in range(3):
            if np.all(board[i] == board[i][0]) and board[i][0] != 0:
                result = board[i][0]
                break
            if np.all(board[:, i] == board[0][i]) and board[0][i] != 0:
                result = board[0][i]
                break
        
        # Check diagonals
        if board[0,0] == board[1,1] == board[2,2] and board[0][0] != 0:
            result = board[0][0]
        elif board[0,2] == board[1,1] == board[2,0] and board[0][2] != 0:
            result = board[0][2]

        if result:
            self.result = Result(result)
            return True

        # Check for tie
        if not (board == 0).any():
            self.result = Result.TIE
            return True

        return False
     
    def is_valid_move(self, move):
        # Checks that the move is a list or tuple of length 2 and that the placement is on an empty tile.
        if type(move) is not list and type(move) is not tuple:
            return False

        if not len(move) == 2:
            return False
         
        if not self.board[move] == 0:
            return False
        
        return True
    
    def get_next_board(self, board, move):
        board = board.copy()
        board[move] = self.turn
        return board

    @staticmethod
    def render(board):
        print(board)

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
    
    def play(self, game_state):
        valid_moves = TicTacToe().get_valid_moves(game_state['board'])
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
    results = {-1:0, 0:0, 1:0}

    for i in range(100):

        #game_state = {'board':np.array([[0,1,2],[0,0,0],[1,1,2]]), 'running':True, 'curr_player':1}
        
        T = TicTacToe()

        p1 = MCTSPlayer("p1", TicTacToe, 100)
        p2 = MCTSPlayer("p2", TicTacToe, 500)

        T.init_players([p1, p2])
        T.run(render=False)

        results[T.result] += 1 
        #print(T.game_state['result'])

    print(results)


def main2():
    for key, val in TicTacToe().__dict__.items():
        print(key)
        print(val)
        print()

def main3():
    agent = AlphaZeroPlayer('p', TicTacToe, 100)
    #agent = MCTSPlayer('p', TicTacToe, 10)
    agent.is_saving_data = True

    t = TicTacToe()
    t.init_players([agent, agent])
    start = time.perf_counter()
    t.run()
    end = time.perf_counter()
    print(end - start)
    #print(t.board)
    #print(agent.saved_data)

def main4():
    folder_path = os.getcwd() + "/saved_models/"
    file_type = r'/*'
    files = glob.glob(folder_path + file_type)
    max_file = max(files, key=os.path.getctime)

    model = 0#tf.keras.models.load_model(max_file)
    
    agent = AlphaZeroPlayer('p', TicTacToe, model, 500)

    t = TicTacToe()
    t.init_players([agent, agent])
    t.run(render=True)

    

if __name__ == '__main__':
    main()