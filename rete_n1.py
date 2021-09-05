import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(42)
n_col = 1
X = np.random.standard_normal((200, n_col))
y = 2 * X + 3

model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=[n_col]) #input_shape = numero colonne
])
#tf.keras.utils.plot_model(model, show_shapes=True)
model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.Adam())
history = model.fit(x=X, y=y, epochs=1000)
y_pred = model.predict(X)
loss_list = history.history['loss']
plt.plot(range(1000), loss_list)
#model.summary()

plt.grid(True)
print(model.layers[0].weights)
#plt.show()