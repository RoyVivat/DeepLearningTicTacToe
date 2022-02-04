from abc import ABC, abstractmethod, abstractproperty
from typing import List

class Player(ABC):
    @abstractmethod
    def play(self, gamestate):
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

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

class TurnBasedGame(Game):
    
    def __init__(self):
        self.gamestate = self.init_gamestate()
        self.running = True

    @abstractmethod
    def init_gamestate(self):
        pass

    @abstractmethod
    def update_gamestate(self, move):
        pass

    def init_players(self, players: List[Player]):
        self.players = players
    
    def get_next_player(self):
        while True:
            for p in self.players:
                yield p

    def run(self):
        turn = self.get_next_player()
        while self.running:
            self.curr_player = next(turn)
            move = self.curr_player.play(self.gamestate)
            self.update_gamestate(move)
            self.render()
            
    
    def render(self):
        raise Exception('Render method not implemented for this game.')

    def save(self):
        raise Exception('Save method not implemented for this game.')

    def load(self):
        raise Exception('Load method not implemented for this game.')



