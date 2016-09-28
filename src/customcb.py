import keras.callbacks
import numpy as np
import math
class SaveBestModel(keras.callbacks.Callback):
    def __init__(self, filepath, verbose=0, thresh=0.005, logs={}):
        super(SaveBestModel, self).__init__()
        self.tr_scores = []
        self.val_scores = []
        self.filepath = filepath
        self.monitor_op = np.greater
        self.best = -np.Inf
        self.verbose = verbose
        self.thresh = thresh

    def on_epoch_end(self, epoch, logs={}):
        self.tr_scores.append(logs.get('acc'))
        self.val_scores.append(logs.get('val_acc'))
        filepath = self.filepath.format(epoch=epoch, **logs)

        if self.monitor_op(self.val_scores[-1], self.best) and math.fabs(self.tr_scores[-1]-self.val_scores[-1])<self.thresh:
            if self.verbose > 0:
                print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s'
                      % (epoch, 'val_acc', self.best,
                         self.val_scores[-1], filepath))
            self.model.save_weights(filepath)
            self.best = self.val_scores[-1]
