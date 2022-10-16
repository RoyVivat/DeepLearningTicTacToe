import tensorflow as tf
from tensorflow.keras import layers


def alphazero_model(input_shape, output_shape):

    def convolutional_layer(x):
        x = input_shape

        return x
    
    def residual_layer(x):
        x = input_shape
        return x

    # Recieve and pad input
    input = layers.Input(input_shape, name="Input")
    x = layers.ZeroPadding2D(input)
    
    # Convolutional block


    # Many resnet blocks


    # Value head
    v = layers.Conv2D(1, (1,1))(x)
    v = layers.Flatten()(v)
    v = layers.BatchNormalization()(v)
    v = layers.ReLU()(v)
    v = layers.Dense(256)(v)
    v = layers.ReLU()(v)
    v = layers.Dense(1)(v)
    value_output = layers.Activation('tanh')(v)

    # Policy head
    p = layers.Conv2D(2, (1,1))(x)
    p = layers.Flatten()(p)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Dense()()
    policy_output = layers.Softmax()(p)

    return tf.keras.Model(input, [value_output, policy_output])

model = alphazero_model((25,25), 0)


model.summary()
tf.keras.utils.plot_model(model, "AlphaZero.png", show_shapes=True)