from abc import abstractmethod

import numpy as np
import tensorflow as tf


class AbstractCritic:
    pass
    # @abstractmethod
    # def __call__(self, r, s, s_next):
    #     """Returns the critics' evaluation of a given state (TD error)."""
# 
    # @abstractmethod
    # def set_eligibility(self, s, value):
    #     """Sets the eligibility for a state to a specified value."""
# 
    # @abstractmethod
    # def reset_eligibility(self):
    #     """Resets the eligibility for all states."""
# 
    # @abstractmethod
    # def update_weights(self, error):
    #     """Updates the eligibility for all states by the specified error."""


class CriticTable(AbstractCritic):
    def __init__(self, env, alpha=0.01, decay_rate=0.9, discount_rate=0.99):
        """
        Har i oppgave å evaluere tilstander.

        Args:
            env:                    Environment, used to fetch state space specifications.
            alpha:                  Critic's learning rate.
            decay_rate:             Trace decay rate (λ) (eligibility decay / decay for prev. states)
            discount_rate:          Discount rate (𝛾) / future decay rate for depreciating future rewards.
        """
        self.env = env
        self.V = {}                 # Format:   state -> value
        self.eligibility = {}
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.discount_rate = discount_rate

    def __call__(self, r, s, s_next):
        """Returns the critics' evaluation of a given state (TD error)."""
        v = self.V.setdefault(s, np.random.uniform(0.05, 0.1))
        v_next = self.V.setdefault(s_next, np.random.uniform(0.05, 0.1))
        self.eligibility.setdefault(s, 0)
        self.eligibility.setdefault(s_next, 0)
        return r + self.discount_rate * v_next - v

    def reset_eligibility(self):
        self.eligibility = {}

    def update_eligibility(self, state):
        for state_, value in self.eligibility.items():
            self.eligibility[state_] = self.discount_rate * self.decay_rate * value
        self.eligibility[state] = 1

    def update_value_func(self, error):
        for state, value in self.eligibility.items():
            self.V[state] += value * self.alpha * error

    def update_all(self, state, error):
        self.update_eligibility(state)
        self.update_value_func(error)


class CriticNetwork(AbstractCritic):
    def __init__(self, env, alpha=0.01, decay_rate=0.9, discount_rate=0.99, layer_dims=None):
        """
        Args:
            env:                    Environment, used to fetch state space specifications.
            alpha:                  Critic's learning rate.
            decay_rate:             Trace decay rate (λ) (eligibility decay / decay for prev. states)
            discount_rate:          Discount rate (𝛾) / future decay rate for depreciating future rewards.
            layer_dims:             Number of units within the models' hidden layers.
        """
        self.env = env
        self.layer_dims = layer_dims
        self.model = self._build_model()
        self._eligibility = [tf.Variable(tf.zeros_like(layer)) for layer in self.model.trainable_weights].copy()
        self.eligibility = self._eligibility.copy()
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.discount_rate = discount_rate

    def __call__(self, r, s, s_next):

        #####################
        ### TODO ###
        # 1. Fix state representation hashable, see if model runs
        # 2. Fix backpropagation decay
        decoded_state = self.env.decode_state(s)
        return self.model(decoded_state)
        #ts = tf.convert_to_tensor(s)
        #tensor_output = self.model(tf.reshape(ts, shape=(1, ts.shape[0])))
        #return tensor_output.numpy().ravel()[0]

    def td_error(self, r, s, s_next):
        """Returns the critics' evaluation of a given state (TD error)."""
        v = self(s)
        v_next = self(s_next)
        return r + self.discount_rate * v_next - v

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=tuple(self.env.get_observation_spec())))
        for units in self.layer_dims:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        model.summary()
        return model

    def reset_eligibility(self):
        self.eligibility = self._eligibility.copy()

    def update_eligibility(self, state):
        ts = self.convert_to_tensor(state)
        weights = self.model.trainable_weights

        # Injection wrapper which records operations such that gradients can be calculated
        with tf.GradientTape() as tape: 
            value = self.model(tf.reshape(ts, shape=(1, len(ts)))) 
        gradients = tape.gradient(value, weights)

        # Update eligibility traces
        for i in range(len(self.eligibility)):
            self.eligibility[i] = self.eligibility[i] *  self.decay_rate
            self.eligibility[i] = tf.math.add(self.eligibility[i], gradients[i])

    def update_value_func(self, error):
        weights = self.model.trainable_weights
        weight_adjustment = [tf.math.multiply(self.eligibility[i], self.alpha, error) for i in range(len(weights))]
        self.model.optimizer.apply_gradients(zip(weight_adjustment, weights))

    def update_all(self, state, error):
        self.update_eligibility(state)
        self.update_value_func(error)
