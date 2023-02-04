import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        self.A = Dense(n_actions, activation=None)
        self.V = Dense(1, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        return V,A
