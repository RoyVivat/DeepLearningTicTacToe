import glob, os


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
        
        x_train, y_train = example_game_data(model, episodes = 100, sim_count = 50)

        model.fit(x_train, y_train, batchSize=32)

        save_model(model)

def example_game_data():
    pass

def main():
    train_loop()


if __name__ == '__main__':
    main()