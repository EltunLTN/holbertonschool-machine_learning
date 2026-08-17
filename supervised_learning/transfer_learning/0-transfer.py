#!/usr/bin/env python3
"""
Trains a CNN to classify CIFAR 10 using transfer learning with a
Keras Application (ResNet50) as a frozen feature extractor.
"""
import tensorflow as tf
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """Pre-processes the data for the model.

    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10
        data, where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels
        for X

    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.resnet50.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    # 1. Load and preprocess the data
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_valid_p, Y_valid_p = preprocess_data(X_valid, Y_valid)

    # 2. Build the frozen feature-extractor: a lambda layer to upscale the
    # 32x32 CIFAR images to a size ResNet50 expects, followed by ResNet50
    # itself (frozen, no top).
    input_shape = (32, 32, 3)
    resize_shape = (224, 224)

    inputs = K.Input(shape=input_shape)
    resized = K.layers.Lambda(
        lambda img: tf.image.resize(img, resize_shape)
    )(inputs)

    base_model = K.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = False

    features = base_model(resized, training=False)
    feature_extractor = K.Model(inputs, features)

    # 3. Precompute the frozen features ONCE for train and validation sets.
    # This is the expensive part (a single forward pass through ResNet50),
    # and doing it once instead of every epoch is what makes training fast.
    print('Extracting training features...')
    train_features = feature_extractor.predict(
        X_train_p, batch_size=256, verbose=1
    )
    print('Extracting validation features...')
    valid_features = feature_extractor.predict(
        X_valid_p, batch_size=256, verbose=1
    )

    # 4. Build and train a small trainable head on top of the precomputed
    # features. This trains in seconds/minutes since it's just a small
    # dense network on fixed-size vectors.
    head = K.Sequential([
        K.Input(shape=train_features.shape[1:]),
        K.layers.Dense(256, activation='relu'),
        K.layers.Dropout(0.4),
        K.layers.Dense(10, activation='softmax')
    ])

    head.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        K.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=2, verbose=1
        ),
        K.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        )
    ]

    head.fit(
        train_features, Y_train_p,
        validation_data=(valid_features, Y_valid_p),
        batch_size=256,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Assemble the FULL end-to-end model (raw 32x32 image in, prediction
    # out) so the saved .h5 is self-contained and usable on raw CIFAR data,
    # then save it compiled as required.
    full_outputs = head(features)
    full_model = K.Model(inputs, full_outputs)

    full_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    val_loss, val_acc = full_model.evaluate(X_valid_p, Y_valid_p, verbose=0)
    print('Final validation accuracy: {:.4f}'.format(val_acc))

    full_model.save('cifar10.h5')
