import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


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
    p = layers.Dense(10)(p)
    policy_output = layers.Softmax()(p)

    return tf.keras.Model(input_layer, [value_output, policy_output])

model = alphazero_model((1,3,3,), 0)
print("Prediction: ", model.predict(np.array([[[[0,0,0],[0,1,0],[0,0,-1]]]])))

model.summary()
tf.keras.utils.plot_model(model, "AlphaZero.png", show_shapes=True)