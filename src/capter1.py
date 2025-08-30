import tensorflow as tf
import numpy as np

input = tf.keras.Input(shape=(1,))
layer_0 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([
    input,
    layer_0
])

model.compile(optimizer='sgd', loss='mean_squared_error')

# define training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict(np.array([10.0])))
print("Here is what I learned: {}".format(layer_0.get_weights()))