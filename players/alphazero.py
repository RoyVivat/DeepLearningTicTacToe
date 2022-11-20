import time
import math
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from mctsplayer import MCTSPlayer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def alphazero_model(input_shape, output_shape):

    def convolutional_layer(input):
        x = layers.Conv2D(256, (3,3), padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x
    
    def residual_layer(input):
        x = convolutional_layer(input)
        x = layers.Conv2D(256, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, input])
        x = layers.ReLU()(x)
        return x

    # Recieve and pad input
    input_layer = layers.Input(shape=input_shape, name="input")
    
    # Convolutional block
    x = convolutional_layer(input_layer)

    # Many resnet blocks
    x = residual_layer(x)
    x = residual_layer(x)

    # Value head
    v = layers.Conv2D(1, (1,1), padding='same')(x)
    v = layers.Flatten()(v)
    v = layers.BatchNormalization()(v)
    v = layers.ReLU()(v)
    v = layers.Dense(256)(v)
    v = layers.ReLU()(v)
    v = layers.Dense(1)(v)
    value_output = layers.Activation('tanh')(v)

    # Policy head
    p = layers.Conv2D(2, (1,1), padding='same')(x)
    p = layers.Flatten()(p)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Dense(output_shape)(p)
    policy_output = layers.Softmax()(p)

    model = tf.keras.Model(input_layer, [value_output, policy_output])
    model.compile(loss=['mean_squared_error', 'categorical_crossentropy'])
    return model

# print('starting code...')
# start_model = time.perf_counter()
# model = alphazero_model((3,3,1,), 9)
# end_model = time.perf_counter()
# print(end_model - start_model)


# time_list = []
# for i in range(20):
#     rand = np.random.rand(1,3,3,1)
#     start = time.perf_counter()
#     res = model(rand)
#     end = time.perf_counter()
#     time_list.append(end-start)
# print(time_list)
# print(np.mean(time_list))

# print(1 + res)
# print(res[1][0][0])

# model.summary()
# tf.keras.utils.plot_model(model, "AlphaZero.png", show_shapes=True)

class AlphaZeroPlayer(MCTSPlayer):
    def __init__(self, name, game, model=None, sim_count=100):
        super().__init__(name, game, sim_count)
        if not model:
            self.model = alphazero_model((3,3,1), 9)
        else:
            self.model = model

    def rollout(self, node):
        ## Performs rollout. Simulates game from current state
        g = self.game(node['state'])
        if g.is_game_over():
            return g.result

        sim_result = self.model(node['state']['board'].reshape(1,3,3,1))[0][0][0]/2 + 0.5
        return sim_result

    def calc_uct(self, node):
        ## Calculates UCT of a node

        if node['visits'] == 0:
            return math.inf
        return node['wins']/node['visits'] + math.sqrt(2) * node['prior'] * math.sqrt(node['parent']['visits'])/(1+node['visits'])


    def backpropogate(self, node, sim_result):
        # Saves all of the win and visit information down the tree path
        while True:
            node['wins'] += sim_result
            node['visits'] += 1
            if not node['parent']:
                break
            
            node = node['parent']

            sim_result = 1-sim_result

    def expand_children(self, node):
        ## Expands the children of a leaf node based on valid moves from that state
        g = self.game(node['state'])
        if g.is_game_over():
            return

        valid_moves = self.game.get_valid_moves(node['state']['board'])
        
        priors = np.random.dirichlet([100 for _ in range(9)], 1).reshape((3,3))
        priors += self.model(node['state']['board'].reshape(1,3,3,1))[1][0].numpy().reshape(3,3)
        priors /= 2

        for move in valid_moves:
            g = self.game(node['state'])
            g.update_game_state(move)
            next_state = g.get_game_state()
            node['children'].append({'wins': 0, 'visits': 0, 'prior': priors[move], 'move': move, 'state': next_state, 'children': [], 'parent': node})