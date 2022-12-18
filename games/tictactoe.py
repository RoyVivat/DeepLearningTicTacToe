import tensorflow as tf
import numpy as np

import sys
sys.path.append('.')
import os
import glob
import logging
from collections import defaultdict

from games.game import TurnBasedGame, Turn, Result
# from players.mcts_player import MCTSPlayer
# from players.az_player import AlphaZeroPlayer
# from players.basic_players import UserTTTPlayer

def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    f = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(f'debug.log', 'w')
    fh.setFormatter(f)
    logger.addHandler(fh)
    return logger
logger = init_logger()
# logger.disabled = True

class TicTacToe(TurnBasedGame):
    INPUT_SIZE = (3,3,2)
    OUTPUT_SIZE = 9

    def __init__(self, game_state = None):
        logger.debug("Initializing TicTacToe.")
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

    def get_valid_moves(self):
        return [tuple(i) for i in np.argwhere(self.board==0)]

    def board2ohe(self, board):
        """Coverts numpy board to one hot encoding input for neural network."""
        return np.stack(np.array([board==val for val in [-1, 1]]), axis=2)

    @staticmethod
    def render(board):
        print(board)
        print()
    
    @staticmethod
    def generate_hashkey(board):
        return board.tobytes()

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

def main2():
    import tensorflow as tf
    from players.az_player import AlphaZeroPlayer
    from players.basic_players import UserTTTPlayer
    
    folder_path = os.getcwd() + "/tictactoe/saved_models/"
    file_type = r'/*'
    files = glob.glob(folder_path + file_type)
    
    print(f'files {files}')
    if not files:
        pass
    else:
        print('loading model...')
        max_file = max(files, key=os.path.getctime)
        print(max_file)
        model = tf.keras.models.load_model(max_file, compile=False)
        model.compile(loss=['mean_squared_error', 'categorical_crossentropy'])
        print('model loaded!')

    p1 = AlphaZeroPlayer(TicTacToe, (3,3,2), 9, model, 10, False)

    p2 = UserTTTPlayer()#TicTacToe, (3,3,2), 9, None, 10, False)

    # results = defaultdict(lambda: 0)

    # for _ in range(100):
    #     O = TicTacToe()
    #     O.init_players([p1,p2])
    #     O.run(render=False)
    #     results[O.result] += 1
    #     p1.mcts.node_dict = {}
    #     p2.mcts.node_dict = {}

    # print(results)

    results = defaultdict(lambda: 0)

    for _ in range(5):
        O = TicTacToe()
        O.init_players([p1,p1])
        O.run(render=True)
        results[O.result] += 1
        p1.mcts.node_dict = {}
        #p2.mcts.node_dict = {}

    print(results)

if __name__ == '__main__':
    main2()