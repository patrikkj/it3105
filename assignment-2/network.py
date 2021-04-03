import tensorflow as tf


class Actor:
    ...


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
        self.optimizer = ActorNetwork.optimizers[optimizer]
        self.activation = ActorNetwork.activations[activation]
        self.alpha = alpha
        self.layer_dims = layer_dims
        self.batch_size = batch_size
        self.model = self._build_model()
    
    @tf.function
    def __call__(self, state):
        """Returns the Actors' evaluation of a given state."""
        decoded = self.env.decode_state(state)
        tensor = tf.convert_to_tensor(decoded)
        reshaped = tf.reshape(tensor, shape=(1, -1))
        return self.model(reshaped)

    def _build_model(self):
        """Builds the Keras mudel used as a state-value approximator."""
        model = tf.keras.Sequential()

        # Add input layer (state encoding)
        input_spec = self.env.spec().observations
        model.add(tf.keras.layers.Input(shape=(input_spec, )))

        # Add hidden layers
        for units in self.layer_dims:
            model.add(tf.keras.layers.Dense(units, activation=self.activation))

        # Add output layer (probability dist. over action space)
        output_spec = self.env.spec().actions
        model.add(tf.keras.layers.Dense(output_spec, activation='softmax'))

        # Add custom transformation layer to modify logits
        # In this manner we might avoid gradient taping!
        # We will need to supply custom imput to the transformation layer
        # TODO: Add custom layer here
        # Might solve this if using tf.sparse functions which normalize
        # using missing values in labels
        
        # Compile model
        model.compile(
            optimizer=self.optimizer(learning_rate=self.alpha),
            loss=tf.keras.losses.SparseCategoricalCrossentropy()
        )
        return model
