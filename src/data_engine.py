import cPickle as pkl
import sys, re, csv
import time
from collections import OrderedDict
import numpy
import scipy.misc as misc
from keras.utils import np_utils

class DataEngine(object):
            
    def __init__(self, mb_size_train, mb_size_test, dataset_path, nb_classes):
        self.mb_size_train = mb_size_train
        self.mb_size_test = mb_size_test
        self.dataset_path = dataset_path
        self.nb_classes = nb_classes

        self.load_data()
    
    def generate_minibatch_idx(self, dataset_size, minibatch_size):
        # generate idx for minibatches SGD
        # output [m1, m2, m3, ..., mk] where mk is a list of indices
        assert dataset_size >= minibatch_size
        n_minibatches = dataset_size / minibatch_size
        leftover = dataset_size % minibatch_size
        idx = range(dataset_size)
        if leftover == 0:
            minibatch_idx = numpy.split(numpy.asarray(idx), n_minibatches)
        else:
            print 'uneven minibath chunking, overall %d, last one %d'%(minibatch_size, leftover)
            minibatch_idx = numpy.split(numpy.asarray(idx)[:-leftover], n_minibatches)
            minibatch_idx = minibatch_idx + [numpy.asarray(idx[-leftover:])]
        minibatch_idx = [idx_.tolist() for idx_ in minibatch_idx]
        return minibatch_idx

    def load_data(self):
        labels = csv.reader(open('../labels.csv'))
        self.data = []
        for row in labels:
            self.data.append(row)

        self.train_split = int(0.7 * len(self.data))
        self.kf_train = self.generate_minibatch_idx(self.train_split, self.mb_size_train)
        self.kf_test = self.generate_minibatch_idx(len(self.data) - self.train_split, self.mb_size_test)

    def prepare_data_train(self, index):
        x , y = []
        for idx in index:
            x.append(misc.imread(data[idx][0] + '.jpg'))
        y = np_utils.to_categorical(y, self.nb_classes)
        return x, y

    def prepare_data_train(self, index):
        x , y = []
        for idx in index:
            x.append(misc.imread(data[idx][0] + '.jpg'))
        y = np_utils.to_categorical(y, self.nb_classes)
        return x, y


def load_data_numpy(folder_path, nb_classes=1, annonation_path=None):
    data_X, data_Y = [], []
    data = csv.reader(open(annonation_path))
    for row in data:   
        data_X.append(misc.imread(folder_path + row[0] + '.jpg'))
        data_Y.append(int(row[1].replace('/r/n', '')))  
    return numpy.asarray(data_X), np_utils.to_categorical(data_Y, nb_classes)

if __name__ == '__main__':
    data_X, data_Y = load_data_numpy('../data/', 10, '../labels.csv')
    print (data_X.shape, data_Y.shape)
    #engine = DataEngine(64, 64, '../data', 10)
