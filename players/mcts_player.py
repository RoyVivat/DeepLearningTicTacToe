import numpy as np

import sys
sys.path.append('.')

from games.game import Player
from players.player_helpers.mcts import MCTS

class MCTSPlayer(Player):
    # TODO: Use of argmax biases the tree expansion (better to randomly select)

    def __init__(self, game, n_simulations=100, is_saving_data=False):
        self.game = game
        self.n_simulations = n_simulations
        self.is_saving_data = is_saving_data
        self.saved_data = []

        self.mcts = MCTS(game)

    def play(self, game_state: dict):
        mcts = self.mcts
        mcts.update_root(game_state)
        mcts.expand_children(mcts.root)

        mcts.mcts(self.n_simulations)

        if self.is_saving_data:
            self.save_data()

        return mcts.get_most_visited_child().move

    def save_data(self): # TODO: Should convert to a data saving factory in the future. Current implementation is for tictactoe only.
        """Saves data for training."""
        visits, moves = zip(*[(child.visits, child.move)
                            for child in self.mcts.root.children])

        # Saves probability distribution and game_board
        sum_visits = sum(visits)
        board = self.game.get_start_board()
        prob_dist = np.zeros(board.flatten().shape)
        for i, move in enumerate(moves):
            prob_dist[board.shape[0]*move[0]+move[1]] = visits[i]/sum_visits
        self.saved_data.append([self.mcts.root.state['board'].astype(
            'float32'), prob_dist.astype('float32')])


def main():
    pass


if __name__ == '__main__':
    main()
