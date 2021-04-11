import tensorflow as tf
from tensorflow.keras import layers

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

    losses = {
        'kl_divergence': tf.keras.losses.KLDivergence(),
        'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy()
    }

    def __init__(self, env,
                 alpha=1e-4,
                 layer_dims=None, 
                 optimizer='adam',
                 activation='relu',
                 loss='categorical_crossentropy',
                 batch_size=32,
                 epochs=5):
        """
        Args:
            env:                    Environment, used to fetch action space specifications.
            alpha:                  Actor's learning rate.
            layer_dims:             Number of units within the models' hidden layers.
            optimizer:              Optimizer to be used for weight updates
            loss:                   Loss function used by the optimizer
            activation:             Activation function for hidden layers
            epochs:                 Number of epochs per learning 
        """
        self.env = env
        self.optimizer = ActorNetwork.optimizers[optimizer]
        self.activation = ActorNetwork.activations[activation]
        self.loss = ActorNetwork.losses[loss]
        self.alpha = alpha
        self.layer_dims = layer_dims
        self.batch_size = batch_size
        self.epochs = epochs
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
            layers.Input(shape=(input_spec, )),                                    # Input layer (state)
            *[layers.Dense(units, self.activation) for units in self.layer_dims],  # Hidden layers
            layers.Dense(output_spec, 'softmax')                                   # Output layer (action)
        ])
        
        model.compile(
            optimizer=self.optimizer(learning_rate=self.alpha),
            loss=self.loss
        )
        return model

    def train(self, x, y):
        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size)
    
    def save(self):
        print("Saving network ...")
    
    def load(self):
        print("Loading network ...")
