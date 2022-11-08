import glob, os
import pandas as pd
from game import Player
from mctsplayer import MCTSPlayer
from tictactoe import TicTacToe

def create_model():
    pass

def save_model(model):
    folder_path = os.getcwd() + "/saved_models/"
    file_type = r'/*txt'
    files = glob.glob(folder_path + file_type)
    max_file = max(files, key=os.path.getctime)

    if not files:
        model.save("saved_models/model0000")
    else:
        model.save("saved_models/model" + str(int(max_file[-8:-4]) + 1).zfill(4))


def train_loop(model = False, training_iters = 80):
    # TODO: I expect more work needs to be done here once example data function is made

    if not model:
        model = create_model()

    for i in range(training_iters):
        
        x_train, y_train = generate_example_game_data(model, episodes = 100, sim_count = 50)

        model.fit(x_train, y_train, batchSize=32)

        save_model(model)

def generate_example_game_data(player: Player, episodes: int, sim_count: int):
        
    agent = MCTSPlayer('player', TicTacToe, sim_count)
    agent.is_saving_data = True

    headers = ['board', 'hist', 'result']
    
    all_train_data = pd.DataFrame(columns=headers)

    for ep in range(episodes):
        t = TicTacToe()
        t.init_players([agent, agent])
        t.run()

        game_data = [elem + [t.result] for elem in agent.saved_data]
        agent.saved_data = []

        game_data = pd.DataFrame(game_data, columns=headers)
        all_train_data = pd.concat([all_train_data, game_data], ignore_index=True)
    
    folder_path = os.getcwd() + "/training_data/"
    file_type = r'/*csv'
    files = glob.glob(folder_path + file_type)

    if not files:
        all_train_data.to_csv(f"{folder_path}train000000")
    else:
        max_file = max(files, key=os.path.getctime)
        all_train_data.to_csv(f"{folder_path}train{int(max_file[-10:-4])+1:06}")

            

def main():
    generate_example_game_data(0, 10, 100)
    #train_loop()


if __name__ == '__main__':
    main()