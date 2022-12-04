from abc import ABC, abstractmethod, abstractproperty
from typing import List
from collections import defaultdict
from enum import IntEnum
import copy

class Player(ABC):
    @abstractmethod
    def play(self, game_info):
        pass

class Turn(IntEnum):
    P1 = 1
    P2 = -1

    def next(turn):
        return Turn.P1 if turn == Turn.P2 else Turn.P2

class Result(IntEnum):
    WIN = 1
    TIE = 0
    LOSS = -1

class TurnBasedGame(ABC):
    
    def __init__(self, game_state = None):
        """Initializes the game. A game state can be given for initialization, otherwise the game will start from scratch.

        Args:
            game_state (dict, optional): Alternate game state to default initialization. Defaults to None.
        """

        if game_state:
            #game_state = copy.deepcopy(game_state)

            for k, v in game_state.items():
                setattr(self, k, v)

            self.board = copy.deepcopy(self.board)

        else:
            self.board = self.get_start_board()
            self.game_history = []
            self.is_saving_history = False
            self.running = True
            self.turn = Turn.P1

    def run(self, render=False):
        """This is the run loop of the game.

        Args:
            render (bool, optional): Print out the game at each move if True. Defaults to False.
        """

        if not self.players:
            raise Exception("Error: Players not initialized")
        
        if self.is_game_over():
            self.running = False

        while self.running:
            if self.is_saving_history:
                self.game_history.append(self.board)

            player = self.get_player()

            move = player.play(self.get_game_state())
            self.update_game_state(move)

            if render:
                self.render(self.board)
    
    def init_players(self, players: List[Player], turn = 0):
        self.players = players
        self.turn = turn

    def get_player(self):
        return self.players[self.turn]
    
    def get_game_state(self):
        return vars(self)

    def update_game_state(self, move):
        # Updates the game based on the last move. Checks for validity, updates the game_state, and checks if the game
        # is over.
        if not self.is_valid_move(move):
            raise Exception("Error: Invalid move:", move)

        self.board = self.get_next_board(self.board, move)
        self.update_player()

        if self.is_game_over():
            self.running = False
    
    def update_player(self):
        self.turn = Turn.next(self.turn)

    def render(self):
        raise Exception('Render method not implemented for this game.')
    
    @staticmethod
    @abstractmethod
    def get_start_board():
        # Gets the initial board state.
        pass

    @abstractmethod
    def is_game_over(self, board):
        # Checks if the game is over
        pass

    @abstractmethod
    def is_valid_move(self, move):
        pass

    @abstractmethod
    def get_next_board(self, board, move):
        pass

    @abstractmethod
    def generate_hashkey(self):
        pass
