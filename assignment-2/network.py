import json

import tensorflow as tf
from tensorflow.keras import layers

from base import Actor

tf.get_logger().setLevel('ERROR')


class ActorNetwork(Actor):
    optimizers = {
        'adam': tf.keras.optimizers.Adam,
        'adagrad': tf.keras.optimizers.Adagrad,
        'rmsprop': tf.keras.optimizers.RMSprop,
        'sgd': tf.keras.optimizers.SGD
    }

    activations = {
        'linear': tf.keras.activations.linear,
        'sigmoid': tf.keras.activations.sigmoid,
        'tanh': tf.keras.activations.tanh,
        'relu': tf.keras.activations.relu
    }

    losses = {
        'kl_divergence': tf.keras.losses.KLDivergence(),
        'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy()
    }

    def __init__(self, env,
                 alpha,
                 alpha_decay,
                 layer_dims, 
                 optimizer,
                 activation,
                 loss,
                 batch_size,
                 epochs, 
                 _episodes=0):
        """
        Args:
            env:                    Environment, used to fetch action space specifications.
            alpha:                  Actor's learning rate.
            alpha_decay:            Actor's learning rate decay.
            layer_dims:             Number of units within the models' hidden layers.
            optimizer:              Optimizer to be used for weight updates
            loss:                   Loss function used by the optimizer
            activation:             Activation function for hidden layers
            epochs:                 Number of epochs per learning 
        """
        self.env = env
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.layer_dims = layer_dims
        self.batch_size = batch_size
        self.epochs = epochs

        self._optimizer = ActorNetwork.optimizers[optimizer]
        self._activation = ActorNetwork.activations[activation]
        self._loss = ActorNetwork.losses[loss]
        self._model = self._build_model()
        self._checkpoint_dir = None     # Set by the serialization method
        self._episodes = _episodes
        
    @tf.function
    def __call__(self, state):
        """Returns the networks evaluation of a given state (observation)."""
        return self._model(state)

    def get_action(self, state):
        return self(state)

    def decay_learning_rate(self):
        _alpha = tf.keras.backend.get_value(self._model.optimizer.learning_rate)
        tf.keras.backend.set_value(self._model.optimizer.learning_rate, _alpha*self.alpha_decay)
        
    def _build_model(self):
        """Builds the Keras mudel used as a state-value approximator."""
        try:
            input_spec = self.env.decode_state(self.env.get_initial_observation()).size
        except:
            input_spec = self.env.spec.observations
        output_spec = self.env.spec.actions

        model = tf.keras.Sequential([
            layers.Input(shape=(input_spec, )),                                     # Input layer (state)
            *[layers.Dense(units, self._activation) for units in self.layer_dims],  # Hidden layers
            layers.Dense(output_spec, 'softmax')                                    # Output layer (action)
        ])
        
        model.compile(
            optimizer=self._optimizer(learning_rate=self.alpha),
            loss=self._loss
        )
        return model

    def train(self, x, y):
        x = self.env.decode_state(x)
        self._model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size)
    
    def save(self, episode=0):
        # Save model parameters
        #tf.keras.models.save_model(self._model, f"{self._checkpoint_dir}/{episode}")
        self._model.save(f"{self._checkpoint_dir}/{episode}")
    

    # -------------------- #
    # Object serialization #
    # -------------------- #
    def serialize(self, agent_dir):
        # Save state of current instance (Exclude private members and env. reference)
        config = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        del config['env']
        with open(f"{agent_dir}/network_config.json", 'w') as f:
            json.dump(config, f, indent=4)
        self._checkpoint_dir = f"{agent_dir}/checkpoints"

    @classmethod
    def from_checkpoint(cls, env, agent_dir, episode):
        config_path = f"{agent_dir}/network_config.json"
        checkpoint_path = f"{agent_dir}/checkpoints/{episode}"

        # Load config and create Network instance
        with open(config_path) as f:
            config = json.load(f)
        obj = cls(env, **config)
        obj._episodes = episode

        # Reload network parameters and recover adaptive learning rate
        obj._model = tf.keras.models.load_model(checkpoint_path)
        alpha = obj.alpha * obj.alpha_decay ** episode 
        tf.keras.backend.set_value(obj._model.optimizer.learning_rate, alpha)
        print(f"Setting alpha to: {alpha}")
        print(f"Successfully loaded ActorNetwork [config={config_path}, checkpoint={checkpoint_path}]")
        return obj
