import cPickle as pkl
import sys, re, csv
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy.misc as misc
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,Adam,Nadam
from keras.layers.advanced_activations import PReLU

def load_data(img_size = 32):
    f = open('../preprocessing/images_list.txt').readlines()
    #print np.asarray(PIL.Image.open("../data/images/"+f[i].strip()))
    X = [img_to_array(load_img("../data/images_resized_" + str(img_size) + "/" + f[i].strip())) for i in range(len(f))]
    X = np.asarray(X).reshape(-1,3,32,32)
    #print f.shape
    Y = pd.read_csv('../data/labels.csv')   
    Y = Y["Label"].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1729, stratify=Y)
    print 'X_train shape:', X_train.shape
    print X_train.shape[0], 'train samples'
    print X_test.shape[0], 'test samples'
    return X_train, X_test, Y_train, Y_test

def prelu_model(img_dim = None, nb_classes = 10):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=img_dim))
    model.add(PReLU())
    model.add(Convolution2D(32, 3, 3))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(PReLU())
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512,init='glorot_uniform'))
    model.add(PReLU())
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes,init='glorot_uniform'))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])
    return model

def simple_model(img_dim = None, nb_classes = 10):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=img_dim))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512,init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes,init='glorot_uniform'))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])

def simple_model_level3(img_dim = None, nb_classes = 10):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=img_dim))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    #model.add(Dense(1024,init='glorot_uniform'))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(512,init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes,init='glorot_uniform'))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])
# if __name__ == '__main__':
#     data_X, data_Y = load_data_numpy('../data/', 10, '../labels.csv')
#     print (data_X.shape, data_Y.shape)
#engine = DataEngine(64, 64, '../data', 10)
