import ssl
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_image, test_labels) = fashion_mnist.load_data()

print('THE TRAIN LABEL', train_labels[0])
#print('THE TRAIN IMAGES', train_images[0])

plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
]) #STUDIA keras.Sequential
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy') #STUDIA compile
model.fit(train_images, train_labels, epochs=5) #STUDIA fit

test_loss = model.evaluate(test_image, test_labels) #STUDIA evaluate
predictions = model.predict(test_image) #STUDIA predict

#plt.show()

print('Done!')
print('Result: ', list(predictions[0]).index(max(predictions[0])))
print('Correct Answer: ', train_labels[0])