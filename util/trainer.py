import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

import sys
sys.path.append('.')
import glob, os
import math
import logging

from players.player_helpers.az_model import alphazero_model
from players.az_player import AlphaZeroPlayer

def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    f = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(f'debug.log', 'w')
    fh.setFormatter(f)
    logger.addHandler(fh)
    return logger
logger = init_logger()

class Trainer():
    def __init__(self, game, game_path):
        logger.debug("Initializing trainer.")
        self.game = game
        self.game_path = game_path

    def save_model(self, model):
        logger.debug('Starting to save model...')
        # Find saved model files
        folder_path = os.getcwd() + self.game_path + 'saved_models/'
        file_type = r'/*'
        files = glob.glob(folder_path + file_type)
        
        # If folder is empty, save a new file else increment id
        if not files:
            logger.debug('Most recent model not found. Saving first model.')
            model.save(folder_path + "model0000")
        else:
            max_file = max(files, key=os.path.getctime)
            logger.debug(f'Most recent model found at {max_file}. Saving new model.')
            model.save(folder_path + 'model' + str(int(max_file[-4:]) + 1).zfill(4))

    def generate_example_game_data(self, model, episodes: int, n_simulations: int):
        logger.debug(f'Generating {episodes} episodes of game data each with {n_simulations} simulations.')

        # Create agent using model
        agent = AlphaZeroPlayer(self.game, self.game.INPUT_SIZE, self.game.OUTPUT_SIZE, model, n_simulations, is_saving_data=True)

        # Create headers for pandas dataframe
        headers = ['board', 'hist', 'result']

        # Initialize characters for loading bar
        sq = chr(9632)
        sp = chr(32)
        
        # Create dataframe
        all_train_data = pd.DataFrame(columns=headers)

        print(f'Progress: [{ascii(254)}]')      # Progress bar
        for ep in range(episodes):
            logger.debug(f'Starting episode {ep}/{episodes}.')
            #Update progress bar
            print(f'\033c')
            print(f'Progress [{sq*int(math.floor(10*(ep+1)/episodes))}{sp*int(math.ceil(10-10*((ep+1)/episodes)))}]')

            # Initialize game with agent playing both sides
            G = self.game()
            G.init_players([agent, agent])
            G.run()

            game_data = [elem + [G.result] for elem in agent.saved_data] # elem consists of [board, hist], so we append result
            for i in range(1, len(game_data), 2):
                game_data[i][0] = np.flip(game_data[i][0], 2)
                game_data[i][2] = -game_data[i][2]
            agent.saved_data = []

            game_data = pd.DataFrame(game_data, columns=headers)
            all_train_data = pd.concat([all_train_data, game_data], ignore_index=True)
        
        folder_path = os.getcwd() + self.game_path + 'training_data/'
        file_type = r'/*'
        files = glob.glob(folder_path + file_type)
        
        print(f'files {files}')
        if not files:
            logger.debug("Most recent training file not found. Creating new training file.")
            with open(folder_path + 'train000000', 'xb') as file:
                pickle.dump(all_train_data, file)
        else:
            max_file = max(files, key=os.path.getctime)
            logger.debug(f'Most recent training file found at {max_file}. Creating new training file.')
            with open(folder_path + f"train{int(max_file[-6:])+1:06}", 'xb') as file:
                pickle.dump(all_train_data, file)

        return all_train_data

    def train_loop(self, model = False, training_iters = 80):

        if not model:

            model = alphazero_model(self.game.INPUT_SIZE, self.game.OUTPUT_SIZE)
            [print(i.shape, i.dtype) for i in model.inputs]
            [print(o.shape, o.dtype) for o in model.outputs]

        for i in range(training_iters):
            
            train = self.generate_example_game_data(model, episodes = 32, n_simulations = 100)

            x_train = tf.convert_to_tensor(train['board'].to_list())

            y_train = [tf.convert_to_tensor(train['result'].to_list()), tf.convert_to_tensor(train['hist'].to_list())]

            model.fit(x_train, y_train, batch_size=32)

            self.save_model(model)
    

def main():
    from games.tictactoe import TicTacToe
    from games.othello import Othello

    folder_path = os.getcwd() + "tictactoe/saved_models/"
    file_type = r'/*'
    files = glob.glob(folder_path + file_type)
    
    trainer = Trainer(TicTacToe, '/tictactoe/')
    
    if not files:
        trainer.train_loop(training_iters=300)
    else:
        max_file = max(files, key=os.path.getctime)
        model = tf.keras.models.load_model(max_file)
        trainer.train_loop(model)

if __name__ == '__main__':
    main()