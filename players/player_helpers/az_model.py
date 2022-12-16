import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import sys
sys.path.append('..')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob

def alphazero_model(input_shape, output_shape):

    def convolutional_layer(input):
        x = layers.Conv2D(256, (3, 3), padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def residual_layer(input):
        x = convolutional_layer(input)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
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
    v = layers.Conv2D(1, (1, 1), padding='same')(x)
    v = layers.Flatten()(v)
    v = layers.BatchNormalization()(v)
    v = layers.ReLU()(v)
    v = layers.Dense(256)(v)
    v = layers.ReLU()(v)
    v = layers.Dense(1)(v)
    value_output = layers.Activation('tanh')(v)

    # Policy head
    p = layers.Conv2D(2, (1, 1), padding='same')(x)
    p = layers.Flatten()(p)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Dense(output_shape)(p)
    policy_output = layers.Softmax()(p)

    model = tf.keras.Model(input_layer, [value_output, policy_output])
    model.compile(loss=['mean_squared_error', 'categorical_crossentropy'])
    
    return model

def alphazero_model_loader(): # Edit to support different model shapes and saving paths
    folder_path = os.getcwd() + "/saved_models/"
    file_type = r'/*'
    files = glob.glob(folder_path + file_type)
    
    print(f'files {files}')
    if not files:
        model = alphazero_model()
    else:
        max_file = max(files, key=os.path.getctime)
        model = tf.keras.models.load_model(max_file)
        return model

def main():
    model = alphazero_model((3,3,2,), 9)
    model.summary()
    tf.keras.utils.plot_model(model, "AlphaZero.png", show_shapes=True)

    
    a = np.array([[0,1,1],[-1,0,1],[-1,-1,0]])
    inp = np.array([(a==val).astype(int) for val in [1, -1]]).reshape((1,3,3,2))
    print(model(inp))
    
if __name__ == '__main__':
    main()