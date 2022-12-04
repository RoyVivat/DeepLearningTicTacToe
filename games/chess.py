import numpy as np

import sys
sys.path.append('.')

from typing import List
from games.game import Player, TurnBasedGame, Turn
from players.player_helpers.mcts import MCTSPlayer

class Chess(TurnBasedGame):
    def __init__(self, game_state = None):
        super().__init__(game_state)
        self.players = {Turn.P1:None, Turn.P2:None}
        self.is_valid_switch = {'r': self.is_valid_rook_move,
                                'n': self.is_valid_knight_move,
                                'b': self.is_valid_bishop_move,
                                'q': self.is_valid_queen_move,
                                'k': self.is_valid_king_move,
                                'p': self.is_valid_pawn_move}
        self.get_valid_switch = {'r': self.get_valid_rook_move,
                                'n': self.get_valid_knight_move,
                                'b': self.get_valid_bishop_move,
                                'q': self.get_valid_queen_move,
                                'k': self.get_valid_king_move,
                                'p': self.get_valid_pawn_move}

    def init_players(self, players):
        self.players = {Turn.P1: players[0], Turn.P2: players[1]}
    
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
        if self.is_king_checked():
            pass

        if not len(self.get_valid_moves(self.board)):
            return True
        
        return False

    def is_king_checked(self):
        pass

    def is_valid_move(self, move):

        # Check for valid indices
        if not self.is_valid_coordinate(move):
            return False
        
        # Check that the start and end colors are appropriate
        if not self.is_valid_coloring(move):
            return False
            
        self.is_valid_bishop_move()
        self.is_valid_rook_move()

    def is_valid_coordinate(self, move):
        """Returns the validity of a move coordinate.

        Args:
            move (list): [start_row, start_col, end_row, end_col]

        Returns:
            bool: 
        """
        if move[0:2] == move[2:4]:
            return False
        if all(m in range(0,8) for m in move):
            return True
        return False

    def is_valid_coloring(self, move):
        """Returns the validity of the color of the start and end squares of a move.

        Args:
            move (list): [start_row, start_col, end_row, end_col]

        Returns:
            bool:
        """
        start, end = move[0:2], move[2:4]
        color = 'w' if self.turn == Turn.P1 else 'b'

        # Check for correct start color
        if not self.board[start][0] == color:
            return False
        
        # Check for correct end color
        if self.board[end][0] == color:
            return False
        
        return True

    def is_valid_rook_move(self, move):
        """Returns the validity of a rook move.
        
        Assumes the correct player and piece is being moved.

        Args:
            move (list): [start_row, start_col, end_row, end_col]

        Returns:
            bool: is valid move.
        """

        start, end = move[0:2], move[2:4]

        # Check for general rook move
        if (not start[0] == end[0]) and (not start[1] == end[1]):
            return False
        
        rowinc = 1 if start[0] < end[0] else -1

    def is_valid_knight_move(self, move):
        start, end = move[0:2], move[2:4]
    
    def is_valid_bishop_move(self, move):
        """Returns the validity of a bishop move.
        
        Assumes the correct player's piece is moving, that the piece is a bishop, and that the end piece is takable.

        Args:
            move (list): [start_row, start_col, end_row, end_col]

        Returns:
            bool: is valid move.
        """
        start, end = move[0:2], move[2:4]

        # Check for general bishop movement
        if (not sum(start) == sum(end)) and (not start[0]-start[1] == end[0]-end[1]):
            return False

        # Increment through every position between the start and end, return false if obstructed
        rowinc = 1 if start[0] < end[0] else -1
        colinc = 1 if start[1] < end[1] else -1

        for r, c in zip(range(start[0]+rowinc, end[0], rowinc), range(start[1]+colinc, end[1], colinc)):
            if not self.board[r,c] == '  ':
                return False
        
        return True

    def get_next_board(self, board, move):
        pass

    @staticmethod
    def render(board):
        print(board)


    def get_valid_moves(self):
        valid_moves = []

        color = 'w' if self.turn == Turn.P1 else 'b'
        for row in self.board:
            for col in row:
                if self.board[row,col][0] == color:
                    valid_moves.append(self.get_valid_switch(self.board[row,col][1]))
    
    def get_valid_rook_move(self, coord):
        pass

    def get_valid_knight_move(self, coord):
        pass

    def get_valid_bishop_move(self, coord):
        pass

    def get_valid_queen_move(self, coord):
        pass

    def get_valid_king_move(self, coord):
        pass

    def get_valid_pawn_move(self, coord):
        pass

def main():
    c = Chess()
    print(c.get_start_board())

if __name__ == '__main__':
    main()