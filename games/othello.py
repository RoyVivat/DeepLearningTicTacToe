import numpy as np
from colorama import Back, Style

import sys
sys.path.append('.')
import time

from games.game import TurnBasedGame, Turn
from players.mcts_player import MCTSPlayer
from players.az_player import AlphaZeroPlayer
from players.basic_players import RandomPlayer, UserTTTPlayer

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
    def generate_hashkey(game_state):
        return game_state['board'].tobytes()

def main():
    start = time.perf_counter()
    O = Othello()
    p1 = AlphaZeroPlayer(Othello, (8,8,1), 64, None, 10, True)
    O.init_players([p1,p1])
    O.run(render=False)
    end = time.perf_counter()

    print(O.result)
    print(f'It took {end - start} seconds to finish the game')


if __name__ == '__main__':
    main()