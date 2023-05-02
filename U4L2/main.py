# Import Statements
import tensorflow as tf
from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

#Load the training and test datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#display one of the digits from X
index = 7698
first_image = train_images[index]
print("Label: ", train_labels[index])
first_image = np.array(first_image, dtype='uint8')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

test_images = test_images.reshape((10000, 28*28))
train_images = train_images.reshape((60000, 28*28))

train_images = train_images / 255
test_images = test_images / 255

before_categ_test_labels = test_labels
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Dense(512, activation='tanh'))
network.add(layers.Dense(256, activation='tanh'))
network.add(layers.Dense(128, activation='tanh'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer = 'rmsprop', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=10, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print (network.summary())
