import numpy as np

import sys
sys.path.append('.')
from dataclasses import dataclass
from collections import deque
import time
import math
import random

from games.game import Player, Turn, Result

class MCTSPlayer(Player):
    # TODO: Use of argmax biases the tree expansion (better to randomly select)

    def __init__(self, name, game, sim_count = 100):
        self.name = name
        self.game = game
        self.is_saving_data = False
        self.saved_data = []
        self.sim_count = sim_count

    def play(self, game_info: dict):
        ## Play loop for MCTS

        # Saves the real state of the game
        self.game_info = game_info

        # Creates the search tree
        # Nodes contain wins, visits, move, state, children, parent
        root = self.Node(wins=0, visits=0, move=None, state=game_info, children=[], parent=None)
        # {'wins': 0,
        #         'visits': 0,
        #         'move': None,
        #         'state': game_info,
        #         'children': [],
        #         'parent': None}

        self.expand_children(root)

        # Runs the MCTS algorithm
        for _ in range(self.sim_count):            
            leaf = self.traverse(root)
            sim_result = self.rollout(leaf)
            self.backpropogate(leaf, sim_result)

        # Finds the best move based on MCTS results
        visits, moves = zip(*[ (child.visits, child.move) for child in root.children ])

        # Saves probability distribution and game_board
        if self.is_saving_data:
            sum_visits = sum(visits)
            prob_dist = np.zeros(9)
            for i, move in enumerate(moves):
                prob_dist[3*move[0]+move[1]] = visits[i]/sum_visits
            self.saved_data.append([game_info['board'].astype('float32'), prob_dist.astype('float32')])
        
        # sum_visits = sum(visits)
        # prob_dist = np.zeros(9)
        # for i, move in enumerate(moves):
        #     prob_dist[3*move[0]+move[1]] = visits[i]/sum_visits
        # print(prob_dist)

        return moves[np.argmax(visits)]
    
    def traverse(self, root):
        ## Finds the next leaf for rollout

        # Start at the root and move down the tree based on UCT until the node doesn't have children

        node = root
        
        while node.children:
            node = self.get_best_uct_child(node)

        self.expand_children(node)

        return node
  
    def expand_children(self, node):
        ## Expands the children of a leaf node based on valid moves from that state
        g = self.game(node.state)
        if g.is_game_over():
            return

        valid_moves = self.game.get_valid_moves(node.state['board'])

        for move in valid_moves:
            g = self.game(node.state)
            g.update_game_state(move)
            next_state = g.get_game_state()
            node.children.append(self.Node(wins=0, visits=0, move=move, state=next_state, children=[], parent=node))

    def get_best_uct_child(self, node):
        ## Gets the best child of a node based on UCT calculations

        scores = []

        for child in node.children:
            scores.append(self.calc_uct(child))
        
        return node.children[np.argmax(scores)]

    def calc_uct(self, node):
        ## Calculates UCT of a node

        if node.visits == 0:
            return math.inf
        return node.wins/node.visits + math.sqrt(2*node.parent.visits/node.visits)

    def rollout(self, node):
        ## Performs rollout. Simulates game from current state

        g = self.game(node.state)
        p1 = self.RandomPlayer('p1', self.game)
        p2 = self.RandomPlayer('p2', self.game)
        g.init_players([p1, p2])
        g.run()
        
        # Checks for the player of interest and updates the result accordingly
        if node.state['turn'] == Turn.P1:
            return (g.result - 1)/-2       # transforms result to win condition (1, 0, -1) -> (0, 0.5, 1)
        return (g.result + 1)/2            # transforms result to win condition (-1, 0, 1) -> (0, 0.5, 1)

    def backpropogate(self, node, sim_result):
        # Saves all of the win and visit information down the tree path
        while True:
            node.wins += sim_result
            node.visits += 1
            if not node.parent:
                break
            
            node = node.parent
            if sim_result == Result.WIN:
                sim_result = Result.LOSS
            elif sim_result == Result.LOSS:
                sim_result = Result.WIN
    
    class RandomPlayer(Player):
        def __init__(self, name, game):
            self.name = name
            self.game = game
            
        def play(self, game_state):
            valid_moves = self.game().get_valid_moves(game_state['board'])
            return valid_moves[np.random.randint(0, len(valid_moves))]

    @dataclass
    class Node():
        wins: int
        visits: int
        move: list
        state: any
        children: list
        parent: any
        
def main():
    pass

if __name__ == '__main__':
    main()