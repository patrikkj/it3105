import tensorflow as tf
from tf.keras.layers import Dense, Input
from base import Actor


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

    def __init__(self, env,
                 alpha=1e-4,
                 layer_dims=None, 
                 optimizer='adam',
                 activation='relu',
                 batch_size=32):
        """
        Args:
            env:                    Environment, used to fetch action space specifications.
            alpha:                  Actor's learning rate.
            layer_dims:             Number of units within the models' hidden layers.
            optimizer:              Optimizer to be used for weight updates
            activation:             Activation function for hidden layers
        """
        self.env = env
        self.optimizer = ActorNetwork.optimizers[optimizer]
        self.activation = ActorNetwork.activations[activation]
        self.alpha = alpha
        self.layer_dims = layer_dims
        self.batch_size = batch_size
        self.model = self._build_model()
    
    @tf.function
    def __call__(self, state):
        """Returns the networks evaluation of a given state (observation)."""
        return self.model(tf.convert_to_tensor(state))

    def get_action(self, state):
        return self(state)

    def _build_model(self):
        """Builds the Keras mudel used as a state-value approximator."""
        input_spec = self.env.spec.observations
        output_spec = self.env.spec.actions

        model = tf.keras.Sequential([
            Input(shape=(input_spec, )),                                    # Input layer (state)
            *[Dense(units, self.activation) for units in self.layer_dims],  # Hidden layers
            Dense(output_spec, 'softmax')                                   # Output layer (action)
        ])

        # Compile model
        model.compile(
            optimizer=self.optimizer(learning_rate=self.alpha),
            loss=tf.keras.losses.KLDivergence()
        )
        return model

    def train(self, x, y):
        self.model.fit(x, y, batch_size=self.batch_size)
    
    def save(self):
        print("Saving network ...")
    
    def load(self):
        print("Loading network ...")
