import tensorflow as tf

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
        decoded = self.env.decode_state(state)
        tensor = tf.convert_to_tensor(decoded)
        reshaped = tf.reshape(tensor, shape=(1, -1))
        return self.model(reshaped)

    def get_action(self, state):
        return self(state)

    def _build_model(self):
        """Builds the Keras mudel used as a state-value approximator."""
        model = tf.keras.Sequential()

        # Add input layer (state encoding)
        input_spec = self.env.spec.observations
        model.add(tf.keras.layers.Input(shape=(input_spec, )))

        # Add hidden layers
        for units in self.layer_dims:
            model.add(tf.keras.layers.Dense(units, activation=self.activation))

        # Add output layer (probability dist. over action space)
        output_spec = self.env.spec.actions
        model.add(tf.keras.layers.Dense(output_spec, activation='softmax'))

        # TODO: Add custom transformation layer to modify logits
        # In this manner we might avoid gradient taping!
        # We will need to supply custom imput to the transformation layer
        # Can assign a custom gradient function to the custom layer
        # Might solve this if using tf.sparse functions which normalize
        # using missing values in labels
        
        # Compile model
        model.compile(
            optimizer=self.optimizer(learning_rate=self.alpha),
            loss=tf.keras.losses.KLDivergence()
        )
        return model

    @tf.function
    def update(self, observations, labels):
        # Wraps tf.ops to record operations such that gradients can be calculated
        with tf.GradientTape() as tape:
            # Forward propagate
            logits = self(observations)
            labels = tf.stop_gradient(labels)

            # TODO: Modify logits by masking prohibited actions

            # Calculate loss
            # TODO: Update using 'self.model.loss' after model.compile(...)
            # From TF source code https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/engine/training.py#L449-L549
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels
            )

        # Compute gradients.
        weights = self.model.trainable_weights
        gradients = tape.gradient(loss, weights)
        self.model.optimizer.apply_gradients(zip(gradients, weights))
        return loss
