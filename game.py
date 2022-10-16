from abc import ABC, abstractmethod, abstractproperty
from typing import List
from collections import defaultdict
import copy

class Player(ABC):
    @abstractmethod
    def play(self, game_info):
        pass

class TurnBasedGame(ABC):
    
    def __init__(self, game_state = None):
        """Initializes the game. A game state can be given for initialization, otherwise the game will start from scratch.

        Args:
            game_state (dict, optional): Alternate game state to default initialization. Defaults to None.
        """

        if game_state:
            game_state = copy.deepcopy(game_state)

            for k, v in game_state.items():
                setattr(self, k, v)

        else:
            self.board = self.get_start_board()
            self.running = True
            self.curr_player = 0 

    @staticmethod
    @abstractmethod
    def get_start_board():
        # Gets the initial board state.
        pass

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
            player = self.get_player()

            move = player.play(self.get_game_state())
            self.update_game_state(move)

            if render:
                self.render(self.board)
    
    def init_players(self, players: List[Player]):
        self.players = players
        self.curr_player = 0

    @abstractmethod
    def is_game_over(self, board):
        # Checks if the game is over
        pass
    
    def get_player(self):
        return self.players[self.curr_player]
    
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

    def is_valid_move(self, move):
        pass

    def get_next_game_state(self, move):
        self.update_board(move)
        self.update_player()
        return self.get_game_state()

    @abstractmethod
    def get_next_board(self, board, move):
        pass    

    def update_player(self):
        self.curr_player = (self.curr_player + 1) % 2

    def render(self):
        raise Exception('Render method not implemented for this game.')