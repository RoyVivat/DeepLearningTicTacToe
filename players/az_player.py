import numpy as np

from players.mcts_player import MCTSPlayer
from players.player_helpers.mcts import AlphaZeroMCTS


class AlphaZeroPlayer(MCTSPlayer):
    def __init__(self, game, model_input_shape, model_output_shape, model=None, n_simulations=100, is_saving_data=False):
        super().__init__(game, n_simulations, is_saving_data)
        self.mcts = AlphaZeroMCTS(game, model_input_shape, model_output_shape, model)

        self.temp = 1

    def play(self, game_state: dict):
        mcts = self.mcts
        mcts.update_root(game_state)
        mcts.expand_children(mcts.root)
        #self.add_dirichlet_noise()

        mcts.mcts(self.n_simulations)

        if self.is_saving_data:
            self.save_data()
        
        mcts.node_dict = {}

        return mcts.get_most_visited_child().move

    def add_dirichlet_noise(self):
        children = self.mcts.root.children
        num_children = len(children)

        dir_noise = np.random.dirichlet(tuple([1 for _ in range(num_children)]), 1)

        for i, child in enumerate(children):
            child.prior = 0.75 * child.prior + 0.25 * dir_noise[0,i]
        