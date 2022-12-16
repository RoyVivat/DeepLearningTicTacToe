import numpy as np
from colorama import Back, Style

import sys
sys.path.append('.')
import time
import glob
import os
from collections import defaultdict

from games.game import TurnBasedGame, Turn
#from players.mcts_player import MCTSPlayer
#from players.az_player import AlphaZeroPlayer
#from players.basic_players import RandomPlayer, UserTTTPlayer

class Othello(TurnBasedGame):
    def __init__(self, game_state = None):
        super().__init__(game_state)
        self.players = {Turn.P1:None, Turn.P2:None}
        self.incs = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
    
    @staticmethod
    def get_start_board():
        # Returns the initial game board
        board = np.zeros([8,8])
        board[3,3], board[4,4] = 1, 1
        board[3,4], board[4,3] = -1, -1
        return board
    
    def init_players(self, players):
        self.players = {1: players[0], -1: players[1]}

    def is_game_over(self):
        # Game ends when there are no legal moves for either player
        if len(self.get_valid_moves()) == 0:
            self.turn = Turn.next(self.turn)
            if len(self.get_valid_moves()) == 0:
                self.turn = Turn.next(self.turn)

                p1 = np.sum(self.board==Turn.P1)
                p2 = np.sum(self.board==Turn.P2)
                if p1 > p2:
                    self.result = Turn.P1
                elif p1 < p2:
                    self.result = Turn.P2
                else:
                    self.result = 0

                return True

        return False
            
    def is_valid_move(self, move):
        # Checks that the move is a list or tuple of length 2 and that the placement is on an empty tile.  
        valid_moves = self.get_valid_moves()
        if len(valid_moves) == 0 and len(move)==0:
            return True
        elif move in valid_moves:
            return True
        
        return False
    
    def get_next_board(self, board, move):
        board = board.copy()
        
        for r_inc, c_inc in self.incs:
            if self.search(move, r_inc, c_inc):
                coord = list(move)
                coord[0] += r_inc
                coord[1] += c_inc
                while not board[coord[0],coord[1]] == self.turn:
                    board[coord[0], coord[1]] = self.turn
                    coord[0] += r_inc
                    coord[1] += c_inc
        
        board[move] = self.turn

        return board

    @staticmethod
    def render(board):
        SQUARE = '\N{BLACK LARGE SQUARE}'
        W_CIRCLE = '\N{MEDIUM WHITE CIRCLE}'
        B_CIRCLE = '\N{MEDIUM BLACK CIRCLE}'

        for i in range(10):
            print(f'{SQUARE}', end='')
        print()
        for row in range(len(board)):
            print(f'{SQUARE}', end='')

            for col in range(len(board[row])):
                if board[row][col] == 1:
                    print(f'{Back.GREEN}{W_CIRCLE}{Style.RESET_ALL}', end='')
                elif board[row][col] == -1:
                    print(f'{Back.GREEN}{B_CIRCLE}{Style.RESET_ALL}', end='')
                else:
                    print(f'{Back.GREEN}  {Style.RESET_ALL}', end='')

            print(f'{SQUARE}', end='')
            print()
        for i in range(10):
            print(f'{SQUARE}', end='')
        print()
        print()

    def get_valid_moves(self):
        zeros = np.argwhere(self.board==0)
        valid = []

        for zero in zeros:
            for r_inc, c_inc in self.incs:
                if self.search(zero, r_inc, c_inc):
                    valid.append(zero)
                    break

        return [tuple(_) for _ in valid]

    def search(self, zero, r_inc, c_inc):
        seen_other = False
        coord = list(zero)
        coord[0] += r_inc
        coord[1] += c_inc
        while coord[0] in range(0,8) and coord[1] in range(0,8):
            if self.board[coord[0], coord[1]] == 0:
                return False
            if not self.board[coord[0], coord[1]] == self.turn:
                seen_other = True
            else:
                if seen_other:
                    return True
                else:
                    return False
            coord[0] += r_inc
            coord[1] += c_inc

    @staticmethod
    def generate_hashkey(board):
        return board.tobytes()

def main():
    import tensorflow as tf
    from players.az_player import AlphaZeroPlayer
    from players.basic_players import UserTTTPlayer
    
    folder_path = os.getcwd() + "/othello/saved_models/"
    file_type = r'/*'
    files = glob.glob(folder_path + file_type)
    
    print(f'files {files}')
    if not files:
        pass
    else:
        print('loading model...')
        max_file = max(files, key=os.path.getctime)
        print(max_file)
        model = tf.keras.models.load_model(max_file)
        print('model loaded!')

    p1 = AlphaZeroPlayer(Othello, (8,8,2), 64, None, 10, False)
    p1.mcts.model = model
    p2 = AlphaZeroPlayer(Othello, (8,8,2), 64, None, 10, False)

    results = defaultdict(lambda: 0)

    for _ in range(10):
        O = Othello()
        O.init_players([p1,p2])
        O.run(render=False)
        results[O.result] += 1
        p1.mcts.node_dict = {}
        p2.mcts.node_dict = {}

    print(results)

    results = defaultdict(lambda: 0)

    for _ in range(10):
        O = Othello()
        O.init_players([p2,p1])
        O.run(render=False)
        results[O.result] += 1

    print(results)

def main2():
    # Negligible: MCTS.update_root(), Othello.generate_hashkey(), 
    # average of 5 runtime for RandomPlayer run(): 0.2620445608 seconds
    # get_valid_moves() on start board:            0.0021001454 seconds and similar on variations
    # AZ play with [1,10,100,500] sims resulted in [0.1409461673, 0.6688825103, 5.985348204, 34.696359343] second times (linear relationship)
    # MCTS.expand_children:                        0.0200577684 seconds
    # Each player.play() run MCTS.expand_children n_simulations+1 times
    # MCTS.rollout():                              0.0208837401 seconds
    # AZ model output:                             0.0166566743 seconds
    # expand_children was reduced from 0.036 to 0.020 so above calculations may be inaccurate

    from players.az_player import AlphaZeroPlayer

    O = Othello()
    p1 = AlphaZeroPlayer(Othello, (8,8,2), 64, None, 10, True)
    O.init_players([p1,p1])
    O.run(render=False)
    print(O.result)
    print(p1.saved_data)


if __name__ == '__main__':
    main2()