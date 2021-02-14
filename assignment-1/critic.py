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

    def __call__(self, state):
        """Returns the critics' evaluation of a given state."""
        self.eligibility.setdefault(state, 0)
        return self.V.setdefault(state, np.random.uniform(0.05, 0.1))

    def td_error(self, r, s, s_next):
        """Returns the temporal difference (TD) error."""
        return r + self.discount_rate * self(s_next) - self(s)

    def reset_eligibility(self):
        self.eligibility = {}

    def update_eligibility(self, state):
        for state_, value in self.eligibility.items():
            self.eligibility[state_] = self.discount_rate * self.decay_rate * value
        self.eligibility[state] = 1

    def update_value_func(self, error):
        for state, value in self.eligibility.items():
            self.V[state] += value * self.alpha * error

    def update_all(self, state, error, is_exploring):
        if is_exploring:
            self.reset_eligibility()
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
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.discount_rate = discount_rate

        # Build model
        self.layer_dims = layer_dims
        self.model = self._build_model()
        self.eligibility = [tf.Variable(tf.zeros_like(weights)) for weights in self.model.trainable_weights]

    def __call__(self, state):
        """Returns the critics' evaluation of a given state."""
        decoded = self.env.decode_state(state)
        tensor = tf.convert_to_tensor(decoded)
        reshaped = tf.reshape(tensor, shape=(1, -1))
        return self.model(reshaped)

    def td_error(self, r, s, s_next):
        """Returns the temporal difference (TD) error."""
        return r + self.discount_rate * self(s_next) - self(s)

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=len(self.env.get_observation())))
        for units in self.layer_dims:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def reset_eligibility(self):
        self.eligibility = [tf.Variable(tf.zeros_like(weights)) for weights in self.model.trainable_weights]

    def update_eligibility(self, state):
        #tensor = tf.convert_to_tensor(state)
        weights = self.model.trainable_weights

        # Wraps tf.ops to record operations such that gradients can be calculated
        with tf.GradientTape() as tape: 
            value = self(state)
        gradients = tape.gradient(value, weights)

        # Update eligibility traces
        for i in range(len(self.eligibility)):
            self.eligibility[i] = self.eligibility[i] *  self.decay_rate
            self.eligibility[i] = tf.math.add(self.eligibility[i], gradients[i])

    def update_value_func(self, error):
        weights = self.model.trainable_weights
        error = tf.reshape(error*1000, -1)
        weight_adjustment = [self.eligibility[i] * error for i in range(len(weights))]
        self.model.optimizer.apply_gradients(zip(weight_adjustment, weights))

        #print(error)
        #before_weights = [w.numpy() for w in weights]
        #print(self.env.get_observation())
        #after_weights = [w.numpy() for w in weights]
        #diffs = [np.abs(w1 - w2).sum() for w1, w2 in zip(before_weights, after_weights)]
        #print(sum(diffs))


    def update_all(self, state, error, is_exploring):
        if is_exploring:
            self.reset_eligibility()
        self.update_eligibility(state)
        self.update_value_func(error)
