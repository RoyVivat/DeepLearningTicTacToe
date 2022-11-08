import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from mctsplayer import MCTSPlayer


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
    input_layer = layers.Input(shape=input_shape, name="Input")
    
    # Convolutional block
    x = convolutional_layer(input_layer)

    # Many resnet blocks
    x = residual_layer(x)
    x = residual_layer(x)

    # Value head
    v = layers.Conv2D(1, (1,1), padding='same')(x)
    #v = layers.Flatten()(v)
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

    return tf.keras.Model(input_layer, [value_output, policy_output])

model = alphazero_model((1,3,3,), 9)
#print("Prediction: ", model.predict(np.array([[[[0,0,0],[0,1,0],[0,0,-1]]]])))
print(model(np.random.rand(1,1,3,3)))

#model.summary()
#tf.keras.utils.plot_model(model, "AlphaZero.png", show_shapes=True)

class AlphaZeroPlayer(MCTSPlayer):
    def __init__(self, name, game):
        super.__init__()
        model = alphazero_model((1,3,3), 10)

    def rollout(self, node):
        ## Performs rollout. Simulates game from current state
        sim_result = self.model(np.array(node['board']).reshape(1,1,3,3))
        return sim_result

    def backpropogate(self, node, sim_result):
        # Saves all of the win and visit information down the tree path
        while True:
            node['wins'] += sim_result
            node['visits'] += 1
            if not node['parent']:
                break
            
            node = node['parent']
            if sim_result == 1:
                sim_result = 0
            elif sim_result == 0:
                sim_result = 1

    def expand_children(self, node):
        ## Expands the children of a leaf node based on valid moves from that state
        g = self.game(node['state'])
        if g.is_game_over():
            return

        valid_moves = self.game.get_valid_moves(node['state']['board'])

        for move in valid_moves:
            g = self.game(node['state'])
            g.update_game_state(move)
            next_state = g.get_game_state()
            node['children'].append({'wins': 0, 'visits': 0, 'move': move, 'state': next_state, 'children': [], 'parent': node})

