# 0-sequential.py

#!/usr/bin/env python3
"""Builds a neural network using the Keras Sequential API."""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a neural network with L2 regularization and dropout."""
    model = K.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(
                K.layers.Dense(
                    units=layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha),
                    input_shape=(nx,)
                )
            )
        else:
            model.add(
                K.layers.Dense(
                    units=layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                )
            )

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model

---

# 10-weights.py

#!/usr/bin/env python3
"""Functions for saving and loading Keras model weights."""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Save a Keras model's weights to a file."""
    network.save_weights(
        filename,
        save_format=save_format
    )


def load_weights(network, filename):
    """Load weights into a Keras model."""
    network.load_weights(filename)

---

# 11-config.py

#!/usr/bin/env python3
"""Functions for saving and loading a Keras model configuration."""

import tensorflow.keras as K


def save_config(network, filename):
    """Save a model's configuration in JSON format."""
    with open(filename, "w") as file:
        file.write(network.to_json())


def load_config(filename):
    """Load a Keras model from a JSON configuration file."""
    with open(filename, "r") as file:
        configuration = file.read()

    return K.models.model_from_json(configuration)

---

# 12-test.py

#!/usr/bin/env python3
"""Evaluates a Keras neural network."""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Evaluate a neural network using testing data."""
    return network.evaluate(
        data,
        labels,
        verbose=verbose
    )

---

# 13-predict.py

#!/usr/bin/env python3
"""Makes predictions using a Keras neural network."""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Return the predictions produced by a neural network."""
    return network.predict(
        data,
        verbose=verbose
    )

---

# 1-input.py

#!/usr/bin/env python3
"""Builds a neural network using the Keras Functional API."""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a functional Keras model with L2 and dropout."""
    inputs = K.Input(shape=(nx,))
    output = inputs

    for i in range(len(layers)):
        output = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(output)

        if i < len(layers) - 1:
            output = K.layers.Dropout(1 - keep_prob)(output)

    return K.Model(inputs=inputs, outputs=output)

---

# 2-optimize.py

#!/usr/bin/env python3
"""Configures Adam optimization for a Keras model."""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Compile a model using Adam and categorical crossentropy."""
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    network.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

---

# 3-one_hot.py

#!/usr/bin/env python3
"""Converts labels into a one-hot encoded matrix."""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Convert a label vector into a one-hot matrix."""
    return K.utils.to_categorical(
        labels,
        num_classes=classes
    )

---

# 4-train.py

#!/usr/bin/env python3
"""Trains a Keras neural network."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descent."""
    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

---

# 5-train.py

#!/usr/bin/env python3
"""Trains a Keras neural network with optional validation data."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """Train a model and optionally analyze validation data."""
    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )

---

# 6-train.py

#!/usr/bin/env python3
"""Trains a Keras model with optional early stopping."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Train a model with optional validation and early stopping."""
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience
        )
        callbacks.append(early_stop)

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

---

# 7-train.py

#!/usr/bin/env python3
"""Trains a Keras model with optional training callbacks."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
    """Train a model with early stopping and learning rate decay."""
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience
        )
        callbacks.append(early_stop)

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            """Calculate the learning rate for the current epoch."""
            return alpha / (1 + decay_rate * epoch)

        learning_rate = K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1
        )
        callbacks.append(learning_rate)

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

---

# 8-train.py

#!/usr/bin/env python3
"""Trains and optionally saves the best Keras model."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """Train a model using optional Keras callbacks."""
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience
        )
        callbacks.append(early_stop)

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            """Calculate the learning rate for the current epoch."""
            return alpha / (1 + decay_rate * epoch)

        learning_rate = K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1
        )
        callbacks.append(learning_rate)

    if save_best and validation_data is not None:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor="val_loss",
            save_best_only=True,
            mode="min"
        )
        callbacks.append(checkpoint)

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

---

# 9-model.py

#!/usr/bin/env python3
"""Functions for saving and loading Keras models."""

import tensorflow.keras as K


def save_model(network, filename):
    """Save an entire Keras model to a file."""
    network.save(filename)


def load_model(filename):
    """Load and return an entire Keras model from a file."""
    return K.models.load_model(filename)

---

