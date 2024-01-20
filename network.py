import tensorflow as tf
from tensorflow import keras
from keras.optimizers.legacy import Adam

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation="relu"),
        keras.layers.Dense(fc2_dims, activation="relu"),
        keras.layers.Dense(n_actions, activation=None)
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")

    return model