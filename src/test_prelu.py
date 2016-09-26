'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

# from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import json
import random

nb_classes = 10

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

def load_data():
    f = open('../preprocessing/images_list.txt').readlines()
    #print np.asarray(PIL.Image.open("../data/images/"+f[i].strip()))
    X = [img_to_array(load_img("../data/images_resized_32/"+f[i].strip())) for i in range(len(f))]
    X = np.asarray(X).reshape(-1,3,32,32)
    #print f.shape
    Y = pd.read_csv('../data/labels.csv')   
    return (X,Y["Label"].tolist())

# the data, shuffled and split between tran and test sets
(X, Y) = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1729, stratify=Y)
print 'X_train shape:', X_train.shape
print X_train.shape[0], 'train samples'
print X_test.shape[0], 'test samples'

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(Y_train, nb_classes)
# Y_test = np_utils.to_categorical(Y_test, nb_classes)
# print Y_train
Y_train = np.array(Y_train)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

for i in range(1,5):
    model = model_from_json(json.loads(open('../results/run_prelu_arch.txt').read()))
    model.load_weights("../results/prelu_best_weights"+"_cv_"+str(i)+".hdf5")
    
    print "Test Accuracy: "
    Y_pred = model.predict_classes(X_test)
    print accuracy_score(Y_test, Y_pred), precision_score(Y_test, Y_pred, average='micro'), recall_score(Y_test, Y_pred, average='micro'), f1_score(Y_test, Y_pred, average='micro')
    print "Train Accuracy: "
    Y_pred = model.predict_classes(X_train)
    print accuracy_score(Y_train, Y_pred), precision_score(Y_train, Y_pred, average='micro'), recall_score(Y_train, Y_pred, average='micro'), f1_score(Y_train, Y_pred, average='micro')