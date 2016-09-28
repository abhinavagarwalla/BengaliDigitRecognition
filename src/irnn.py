'''This is a reproduction of the IRNN experiment
with pixel-by-pixel sequential MNIST in
"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton
arXiv:1504.00941v2 [cs.NE] 7 Apr 2015
http://arxiv.org/pdf/1504.00941v2.pdf
Optimizer is replaced with RMSprop which yields more stable and steady
improvement.
Reaches 0.93 train/test accuracy after 900 epochs
(which roughly corresponds to 1687500 steps in the original paper.)
'''
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.initializations import normal, identity
from keras.optimizers import RMSprop, Nadam, SGD
from keras.utils import np_utils
import numpy as np
import pandas as pd
import json
import random
from sklearn.cross_validation import train_test_split
import json
from data_model import load_data

batch_size = 256
nb_classes = 10
nb_epochs = 1500
hidden_units = 100

learning_rate = 1e-5
clip_norm = 1.0

# images are RGB
img_channels = 3
img_size = 32
random.seed(1729)

X_train, X_test, Y_train, Y_test = load_data(img_size)

X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print 'X_train shape:', X_train.shape
print X_train.shape[0], 'train samples'
print X_test.shape[0], 'test samples'

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

print 'Evaluate IRNN...'
model = Sequential()
model.add(SimpleRNN(output_dim=hidden_units,
                    init=lambda shape, name: normal(shape, scale=0.001, name=name),
                    inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),
                    activation='relu',
                    input_shape=X_train.shape[1:]))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=2, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, verbose=0)
print 'IRNN test score:', scores[0]
print 'IRNN test accuracy:', scores[1]

model.save_weights('../results/irnn_wieghts')
with open('../results/irnn_arch.txt','w') as outfile:
    json.dump(model.to_json(), outfile)
 
