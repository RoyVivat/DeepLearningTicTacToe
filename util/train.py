import glob, os
import pandas as pd
import math
import tensorflow as tf
import numpy as np
from games.game import Player
from players.mctsplayer import MCTSPlayer
from games.tictactoe import TicTacToe
from players.alphazero import AlphaZeroPlayer, alphazero_model

def save_model(model):
    folder_path = os.getcwd() + "/saved_models/"
    file_type = r'/*'
    files = glob.glob(folder_path + file_type)
    
    print(f'files {files}')
    if not files:
        model.save("saved_models/model0000")
    else:
        max_file = max(files, key=os.path.getctime)
        model.save("saved_models/model" + str(int(max_file[-4:]) + 1).zfill(4))

def generate_example_game_data(model, episodes: int, sim_count: int):
        
    agent = AlphaZeroPlayer('player', TicTacToe, model, sim_count)
    agent.is_saving_data = True

    headers = ['board', 'hist', 'result']

    sq = chr(9632)
    sp = chr(32)
    
    all_train_data = pd.DataFrame(columns=headers)
    print(f'Progress: [{ascii(254)}]')
    for ep in range(episodes):
        print(f'\033c')
        print(f'Progress [{sq*int(math.floor(10*(ep+1)/episodes))}{sp*int(math.ceil(10-10*((ep+1)/episodes)))}]')

        t = TicTacToe()
        t.init_players([agent, agent])
        t.run()

        game_data = [elem + [t.result] for elem in agent.saved_data]
        for i in range(1, len(game_data), 2):
            game_data[i][0] = -game_data[i][0]
            game_data[i][2] = -game_data[i][2]
        agent.saved_data = []

        game_data = pd.DataFrame(game_data, columns=headers)
        all_train_data = pd.concat([all_train_data, game_data], ignore_index=True)
    
    folder_path = os.getcwd() + "/training_data/"
    file_type = r'/*'
    files = glob.glob(folder_path + file_type)
    
    print(f'files {files}')
    if not files:
        all_train_data.to_csv(f"{folder_path}train000000")
    else:
        max_file = max(files, key=os.path.getctime)
        all_train_data.to_csv(f"{folder_path}train{int(max_file[-6:])+1:06}")

    return all_train_data

def train_loop(model = False, training_iters = 80):
    # TODO: I expect more work needs to be done here once example data function is made

    if not model:
        model = alphazero_model((3,3,1), 9)
        [print(i.shape, i.dtype) for i in model.inputs]
        [print(o.shape, o.dtype) for o in model.outputs]

    for i in range(training_iters):
        
        train = generate_example_game_data(model, episodes = 128, sim_count = 100)

        x_train = tf.convert_to_tensor(train['board'].to_list())

        y_train = [tf.convert_to_tensor(train['result'].to_list()), tf.convert_to_tensor(train['hist'].to_list())]

        model.fit(x_train, y_train, batch_size=32)

        save_model(model)
 

def main():
    #generate_example_game_data(0, 10, 100)
    folder_path = os.getcwd() + "/saved_models/"
    file_type = r'/*'
    files = glob.glob(folder_path + file_type)
    
    print(f'files {files}')
    if not files:
        train_loop()
    else:
        max_file = max(files, key=os.path.getctime)
        model = tf.keras.models.load_model(max_file)
        train_loop(model)


if __name__ == '__main__':
    main()