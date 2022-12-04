from players.mcts_player import MCTSPlayer
from players.player_helpers.mcts import AlphaZeroMCTS


class AlphaZeroPlayer(MCTSPlayer):
    def __init__(self, game, model_input_shape, model_output_shape, model=None, n_simulations=100, is_saving_data=False):
        super().__init__(game, n_simulations, is_saving_data)
        self.mcts = AlphaZeroMCTS(game, model_input_shape, model_output_shape, model)

        self.temp = 1