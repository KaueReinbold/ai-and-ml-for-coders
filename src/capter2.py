import tensorflow as tf

desired_accuracy = 0.95


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    """
    Custom callback that stops training when accuracy reaches 95%.

    This prevents overfitting and saves computational time by automatically
    terminating training once the desired accuracy threshold is achieved.
    """

    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > desired_accuracy):
            print(
                f"\nReached {desired_accuracy * 100}% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = EarlyStoppingCallback()

data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    # Flatten (input_shape=(28, 28))
    # Purpose: Converts 2D image data to 1D for Dense layers
    # How: Transforms 28x28 pixel grid into 784-length vector
    # Result: Bridge between image input and neural network processing
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Dense (256, activation=tf.nn.relu)
    # Purpose: Fully connected layer for pattern learning
    # How: Each neuron connects to all previous layer outputs
    # Result: Learns complex features through weighted connections
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Dropout (0.2)
    # Purpose: Prevents overfitting
    # How: Randomly turns off 20% of neurons during training
    # Result: Model learns more robust, generalizable patterns
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    # 10-Class Problem
    # Fashion-MNIST: 10 clothing categories (0-9)
    # Final layer: 10 neurons with softmax → probabilities for each class
    # Labels: Simple integers (0, 1, 2... 9)
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    # Job: Adjusts model weights to reduce loss
    # Special: Adaptive learning rate + momentum
    # Why popular: Works well out-of-the-box, self-adjusting
    optimizer='adam',
    # Sparse Categorical Crossentropy
    # Loss function: Measures "how wrong" predictions are
    # "Sparse": Works with integer labels (not one-hot encoded)
    # Goal: Minimize this value during training
    loss='sparse_categorical_crossentropy',
    # Accuracy Metric
    # What: Percentage of correct predictions
    # Purpose: Easy-to-understand progress tracker
    # Example: 85% = got 85 out of 100 images right
    metrics=['accuracy']
)

model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])

# Tests the trained model on new, unseen data (test_images)
# Calculates final performance metrics (loss and accuracy)
# No learning happens - just pure evaluation
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
