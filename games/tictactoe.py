import tensorflow as tf
import numpy as np

import sys
sys.path.append('.')
import os
import absl.logging
import glob

from games.game import TurnBasedGame, Turn, Result
from players.mcts_player import MCTSPlayer
from players.az_player import AlphaZeroPlayer
from players.basic_players import UserTTTPlayer

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
        print()

    def get_valid_moves(self):
        return [tuple(i) for i in np.argwhere(self.board==0)]

    @staticmethod
    def generate_hashkey(game_state):
        return game_state['board'].tobytes()

def main():
    results = {-1:0, 0:0, 1:0}

    folder_path = os.getcwd() + "/saved_models/"
    file_type = r'/*'
    files = glob.glob(folder_path + file_type)
    max_file = max(files, key=os.path.getctime)

    model = tf.keras.models.load_model(max_file)


    for i in range(15):

        T = TicTacToe()
        print(T.board)

        p1 = MCTSPlayer(TicTacToe, 100)
        p2 = AlphaZeroPlayer(TicTacToe, model, 100)

        T.init_players([p1, p2])
        T.run(render=False)

        results[T.result] += 1 
        #print(T.game_state['result'])

    print(results)

# def main3():
#     agent = AlphaZeroPlayer('p', TicTacToe, 100)
#     #agent = MCTSPlayer('p', TicTacToe, 10)
#     agent.is_saving_data = True

#     t = TicTacToe()
#     t.init_players([agent, agent])
#     start = time.perf_counter()
#     t.run()
#     end = time.perf_counter()
#     print(end - start)
#     #print(t.board)
#     #print(agent.saved_data)

# def main4():
#     folder_path = os.getcwd() + "/saved_models/"
#     file_type = r'/*'
#     files = glob.glob(folder_path + file_type)
#     max_file = max(files, key=os.path.getctime)

#     model = tf.keras.models.load_model(max_file)
    
#     agent = AlphaZeroPlayer('p', TicTacToe, model, 100)
#     agent.is_saving_data = True
#     p2 = MCTSPlayer('p2', TicTacToe, 100)
#     t = TicTacToe()
#     t.init_players([p2, agent])
#     t.run(render=True)

#     print(agent.saved_data)

    

if __name__ == '__main__':
    main()