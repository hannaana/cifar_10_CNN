# example of loading the cifar10 dataset
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import tensorflow as tf

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
# summarize loaded dataset
print("Train: X={}, y={}".format(trainX.shape, trainy.shape))
print('Test: X={}, y={}'.format(testX.shape, testy.shape))
# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(trainX[i])
# show the figure
plt.show()

# one hot encode target values
trainY = tf.keras.utils.to_categorical(trainy)
testY = tf.keras.utils.to_categorical(testy)

# convert from integers to floats
trainX_norm = trainX.astype('float32')
testX_norm = testX.astype('float32')
# normalize to range 0-1
trainX_norm = trainX_norm / 255.0
testX_norm = testX_norm / 255.0
model = tf.keras.Sequential()
