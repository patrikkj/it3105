from abc import abstractmethod
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.random.set_seed(1)
tf.get_logger().setLevel('ERROR')


class Critic:
    # Critic types
    TABLE = "table"
    NETWORK = "network"

    def __init__(self, env, alpha, decay_rate, discount_rate, reset_on_explore):
        self.env = env
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.discount_rate = discount_rate
        self.reset_on_explore = reset_on_explore
    
    def td_error(self, reward, state, state_next):
        """
        Returns the temporal difference (TD) error.
             𝛿 = rₜ₊₁ + γ * Vθ(sₜ₊₁) − Vθ(sₜ)
        """
        return reward + self.discount_rate * self(state_next) - self(state)

    @abstractmethod
    def __call__(self, state):
        ...

    @abstractmethod
    def reset_eligibility(self):
        ...

    @abstractmethod
    def update_eligibility(self, state):
        ...

    @abstractmethod
    def update_weights(self, error):
        ...

    @abstractmethod
    def update(self, state, error, is_exploring):
        ...

    @staticmethod
    def from_type(type_, *args, **kwargs):
        if type_ == Critic.TABLE:
            return CriticTable(*args, **kwargs)
        elif type_ == Critic.NETWORK:
            return CriticNetwork(*args, **kwargs)


class CriticTable(Critic):
    def __init__(self, env,
                 alpha=0.01,
                 decay_rate=0.9,
                 discount_rate=0.99,
                 reset_on_explore=True,
                 **kwargs):
        """
        Har i oppgave å evaluere tilstander.

        Args:
            env:                    Environment, used to fetch state space specifications.
            alpha:                  Critic's learning rate.
            decay_rate:             Trace decay rate (λ) (eligibility decay / decay for prev. states)
            discount_rate:          Discount rate (𝛾) / future decay rate for depreciating future rewards.
        """
        super().__init__(env, alpha, decay_rate, discount_rate, reset_on_explore)
        self.V = {}
        self.eligibility = {}

    def __call__(self, state):
        """Returns the critics' evaluation of a given state."""
        self.eligibility.setdefault(state, 0)
        return self.V.setdefault(state, np.random.uniform(0.05, 0.1))

    def reset_eligibility(self):
        self.eligibility = {}

    def update_eligibility(self, state):
        for state_, value in self.eligibility.items():
            self.eligibility[state_] = self.discount_rate * self.decay_rate * value
        self.eligibility[state] = 1

    def update_weights(self, error):
        for state, value in self.eligibility.items():
            self.V[state] += value * self.alpha * error

    def update(self, state, error, is_exploring):
        if is_exploring and self.reset_on_explore:
            self.reset_eligibility()
        self.update_eligibility(state)
        self.update_weights(error)


class CriticNetwork(Critic):
    def __init__(self, env,
                 alpha=0.01,
                 decay_rate=0.9,
                 discount_rate=0.99,
                 reset_on_explore=True, 
                 layer_dims=None, 
                 **kwargs):
        """
        Args:
            env:                    Environment, used to fetch state space specifications.
            alpha:                  Critic's learning rate.
            decay_rate:             Trace decay rate (λ) (eligibility decay / decay for prev. states)
            discount_rate:          Discount rate (𝛾) / future decay rate for depreciating future rewards.
            layer_dims:             Number of units within the models' hidden layers.
        """
        super().__init__(env, alpha, decay_rate, discount_rate, reset_on_explore)

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

    def _build_model(self):
        """Builds the Keras mudel used as a state-value approximator."""
        model = tf.keras.Sequential()
        input_spec = len(self.env.decode_state(*self.env.get_observation()))
        model.add(tf.keras.layers.Input(shape=(input_spec, )))
        for units in self.layer_dims:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.alpha)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def reset_eligibility(self):
        self.eligibility = [tf.Variable(tf.zeros_like(weights)) for weights in self.model.trainable_weights]

    def update_eligibility(self, state):
        # Wraps tf.ops to record operations such that gradients can be calculated
        with tf.GradientTape() as tape:
            value = self(state)
        gradients = tape.gradient(value, self.model.trainable_weights)

        # Applies the eligibility update rule for Semi-Gradient Descent with TD.
        self.eligibility = [e * self.discount_rate * self.decay_rate for e in self.eligibility]
        self.eligibility = [e + dW for e, dW in zip(self.eligibility, gradients)]

    def update_weights(self, error):
        """
        Applies the weight update rule:
            w = w + learning_rate * error * e
        """
        weights = self.model.trainable_weights
        error = tf.reshape(-error, -1)
        weight_update = [error * e for e in self.eligibility]
        self.model.optimizer.apply_gradients(zip(weight_update, weights))

    def update(self, state, error, is_exploring):
        if is_exploring and self.reset_on_explore:
            self.reset_eligibility()
        self.update_eligibility(state)
        self.update_weights(error)
