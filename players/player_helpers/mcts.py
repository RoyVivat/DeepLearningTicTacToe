import numpy as np

import sys
sys.path.append('.')
import math
from dataclasses import make_dataclass, field

from games.game import Turn, TurnBasedGame
from players.basic_players import RandomPlayer
from players.player_helpers.az_model import alphazero_model


class MCTS():
    """Represents a generic MCTS algorithm."""

    def __init__(self, game: TurnBasedGame, node_fields = [], selection_algorithm = None):
        """Constructs all the necessary attributes for the MCTS algorithm.

        Args:
            node_fields (List[Tuple[str, type]]): Specify additional node fields for Node class.
        """

        self.game = game

        # Initializes a Node dataclass
        default_node_fields = [('wins', float, 0), ('visits', int, 0), ('move', any, None),
                               ('state', any, None), ('children', list, field(default_factory=list)), ('parent', any, None)]
        default_node_fields.extend(node_fields)
        self.Node = make_dataclass('Node', default_node_fields)
        self.node_dict = {}

        self.root = None

        self.selection_algorithm = selection_algorithm if selection_algorithm else self.calc_uct

    def update_root(self, game_state):
        """Updates the root node to the new state of interest."""
        hashkey = self.game.generate_hashkey(game_state)
        
        if hashkey in self.node_dict:
            self.root = self.node_dict[hashkey]
        else:
            self.root = self.Node()
            self.root.state = game_state
            self.node_dict[hashkey] = self.root
    
    def expand_children(self, node):
        """Expands the children of a leaf node based on valid moves from that state."""

        g = self.game(node.state)
        if g.is_game_over():
            return

        valid_moves = g.get_valid_moves()

        for move in valid_moves:
            g = self.game(node.state)
            g.update_game_state(move)
            next_state = g.get_game_state()
            hashkey = self.game.generate_hashkey(next_state)
            next_node = self.Node(wins=0, visits=0, move=move, state=next_state, children=[], parent=node)
            self.node_dict[hashkey] = next_node
            node.children.append(next_node)

    def mcts(self, n_simulations: int):
        """Does n cycles of the mcts algorithm."""
        for _ in range(n_simulations):
            leaf = self.traverse()
            sim_result = self.rollout(leaf)
            self.backpropogate(leaf, sim_result)

    
    def traverse(self):
        """Finds the next leaf for rollout."""

        node = self.root

        while node.children:
            node = self.get_best_child(node)

        self.expand_children(node)

        return node

    def get_best_child(self, node):
        """Gets the best child based on the UCT algorithm.
        """
        scores = []

        for child in node.children:
            scores.append(self.selection_algorithm(child))

        return node.children[np.argmax(scores)]

    def calc_uct(self, node):
        """Calculates UCT of a node."""

        if node.visits == 0:
            return math.inf
        return node.wins/node.visits + math.sqrt(2*node.parent.visits/node.visits)

    def rollout(self, node):
        """Performs rollout. Simulates game from current state."""

        g = self.game(node.state)
        p1, p2 = RandomPlayer(self.game), RandomPlayer(self.game)
        g.init_players([p1, p2])
        g.run()

        # Checks for the player of interest and updates the result accordingly
        # We reverse the output when player 2 made the last move
        if node.state['turn'] == Turn.P1:
            # Transforms result to win condition (1, 0, -1) -> (0, 0.5, 1)
            return (g.result - 1)/-2
        # Transforms result to win condition (-1, 0, 1) -> (0, 0.5, 1)
        return (g.result + 1)/2

    def backpropogate(self, node, sim_result):
        """Saves all of the win and visit information down the tree path."""

        while True:
            node.wins += sim_result
            node.visits += 1
            if not node.parent:
                break

            node = node.parent
            sim_result = 1-sim_result


    def get_most_visited_child(self): # TODO: May be more useful to switch to move instead of child
        """Returns the most visited node after running the mcts algorithm.

        Returns:
            Node: Object containing relevant game information
        """
        children, visits = zip(*[(child, child.visits)
                            for child in self.root.children])

        return children[np.argmax(visits)]

class AlphaZeroMCTS(MCTS):
    def __init__(self, game, model_input_shape, model_output_shape, model):
        super().__init__(game, [('prior', float, 1)], self.calc_puct)

        if not model:
            self.model = alphazero_model(model_input_shape, model_output_shape)
        else:
            self.model = model
        
        self.temp = 1

    def rollout(self, node):
        """Performs rollout. Uses AlphaZero value output instead of simulation."""
        g = self.game(node.state)
        if g.is_game_over():
            result = (g.result+1)/2
            return result if node.state['turn'] == Turn.P2 else 1-result

        # Change board perspective for model
        board = -node.state['board'] if node.state['turn'] == Turn.P2 else node.state['board']

        sim_result = (-self.model(board.reshape(tuple([1] + list(self.model.input_shape)[1:])))[0][0][0]+1)/2
        return sim_result

    def calc_puct(self, node):
        """Calculates PUCT of a node with a prior probability."""
        if node.visits == 0:
            return math.inf
        return node.wins/node.visits + math.sqrt(self.temp) * node.prior * math.sqrt(node.parent.visits)/(1+node.visits)

    def expand_children(self, node):
        """Expands the children of a leaf node based on valid moves from that state and prior probabilities of its children."""
        g = self.game(node.state)
        if g.is_game_over():
            return

        valid_moves = g.get_valid_moves()
        priors = self.model(node.state['board'].reshape(tuple([1] + list(self.model.input_shape)[1:])))[1][0].numpy().reshape(g.board.shape)

        for move in valid_moves:
            g = self.game(node.state)
            g.update_game_state(move)
            next_state = g.get_game_state()
            hashkey = self.game.generate_hashkey(next_state)
            next_node = self.Node(wins=0, visits=0, move=move, state=next_state, children=[], parent=node, prior=priors[move])
            self.node_dict[hashkey] = next_node
            node.children.append(next_node)
