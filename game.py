from abc import ABC, abstractmethod, abstractproperty
from typing import List
from collections import defaultdict
import copy

class Player(ABC):
    @abstractmethod
    def play(self, game_info):
        pass


class Game(ABC):

    @abstractmethod
    def init_players(self, players: List[Player]):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def render(self):
        pass

class TurnBasedGame(Game):
    
    def __init__(self, game_state = None):
        """Initializes the game. A game state can be given for initialization, otherwise the game will start from scratch.

        Args:
            game_state (dict, optional): Alternate game state to default initialization. Defaults to None.
        """

        if game_state:
            self.game_state = copy.deepcopy(game_state)
            #for key in self.game_state:
            #    self.game_state[key] = self.game_state[key].copy()

        else:
            self.init_game_state()

    def init_game_state(self):
        # Initialized the game_state. Game_state should contain information about the board state, if the game is running,
        # the players, whose turn it is, optionally the game history, and any information that is necessary to track in
        # the game.
        self.game_state = defaultdict(lambda: None, {'board': self.get_start_board(), 'running': True, 'players': None, 'curr_player': 0})

    @abstractmethod
    def get_start_board():
        # Gets the initial board state.
        pass

    def update_game_state(self, move):
        # Updates the game based on the last move. Checks for validity, updates the game_state, and checks if the game
        # is over.
        if not self.is_valid_move(move):
            raise Exception("Error: Invalid move:", move)

        self.game_state['board'] = self.get_next_board(self.game_state['board'], move)
        self.update_player()


        if self.is_game_over():
            self.game_state['running'] = False
            
    @abstractmethod
    def is_valid_move(self, move):
        pass
    
    def get_next_game_state(self, move):
        self.update_board(move)
        self.update_player()
        return self.game_state

    @staticmethod
    @abstractmethod
    def get_next_board(self, board, move):
        pass

    def init_players(self, players: List[Player]):
        self.game_state['players'] = players
        self.game_state['curr_player'] = 0
    
    def get_player(self):
        return self.game_state['players'][self.game_state['curr_player']]

    def update_player(self):
        self.game_state['curr_player'] = (self.game_state['curr_player'] + 1) % 2

    def run(self, render=False):
        if not self.game_state['players']:
            raise Exception("Error: Players not initialized")
        
        if self.is_game_over():
            self.game_state['running'] = False

        while self.game_state['running']:
            player = self.get_player()

            move = player.play(self.game_state)
            self.update_game_state(move)

            if render:
                self.render()
            
    def render(self):
        raise Exception('Render method not implemented for this game.')