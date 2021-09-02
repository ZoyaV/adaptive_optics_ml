import pickle
import numpy as np
import torch
import cv2

from google.colab import drive
drive.mount('/content/gdrive')

import os
import sys
py_file_location = '/content/gdrive/My Drive/ao_prediction/notebooks'
sys.path.append(os.path.abspath(py_file_location))

from training_seting import opt


def preprocessing(X):
   
    X = np.asarray(X)
    X_train = torch.from_numpy(X).float()
    return X_train


class WavefrontData():
    SLICE = 10
    IMG_SIZE = 128

    def __init__(self, path, shuffle=False, type_f = "SCREEN", batch_size=20, data_last=15, data_ahead=5, axis=0, test=False,
                 alternative=None):
        self.path = path
        self.type_f = type_f
        self.index = 0
        self.X_train = []
        self.y_train = []
        self.y_train_dat = []
        self.batch_size = opt.batch_size
        self.data_last = opt.ahead
        self.data_ahead = opt.predict
        self.axis = axis
        self.test = False
        self.data = []
        self.alternative_data = None
        if alternative is not None:
            self.load_from_source(alternative)
        else:
            self.__load()
    def dropna(self,X):
        X_new = []
        for x in X:
          img = cv2.resize(x[self.SLICE:-self.SLICE, self.SLICE:-self.SLICE], (self.IMG_SIZE,self.IMG_SIZE))
          if not np.sum(np.isnan(x)):
            X_new.append(img)
        return X_new

    def __load(self):
        with open(self.path, 'rb') as f:
            screen, img_2, sum_phase, Y = pickle.load(f)
        if self.type_f == "SCREEN":
            X_train = screen
        if self.type_f == "PHASE":
            X_train = sum_phase
        dropnaX = self.dropna(X_train)
        
        self.X_train = preprocessing(dropnaX)
        self.data = self.X_train

    def normalizate_pics(self):
        indx = []
        # self.y_train = (self.y_train - torch.min(self.y_train))/(torch.max(self.y_train) - torch.min(self.y_train))
        for i in range(len(self.data)):
            self.data[i] = (self.data[i] - torch.min(self.data[i])) / (
                        torch.max(self.data[i]) - torch.min(self.data[i]))
            # print(torch.min(self.y_train[i,0]))
            if ((torch.mean(self.data[i, 0]) != torch.max(self.data[i, 0]))
                    and torch.sum(torch.isnan(self.data[i, 0]).int()) == 0):
                indx.append(i)
        self.data = self.data[indx]
        return self.data

    def load_from_source(self, source):
        self.alternative_data = source

    def dataset(self):
        return self.data

    def reshaped(self):
        if self.axis == 0:
            self.normalizate_pics()
            self.data = self.data

            size = self.data_last + self.data_ahead
            splits = np.array_split(self.data.detach().numpy(), size, axis=0)
            #  print(self.data)
            self.data = np.asarray(splits).reshape(-1, size, self.IMG_SIZE, self.IMG_SIZE, 1)
            indexes = list(range(self.data.shape[0]))
            np.random.shuffle(indexes)
            self.data = self.data[indexes]
            # for i in range(data.shape[0]):
            #   data[i] = (data[i] - np.min(data[i]))/(np.max(data[i]) - np.min(data[i]))
            self.data = np.asarray(np.array_split(self.data, self.batch_size, axis=0)).reshape(-1, self.batch_size,
                                                                                               size, self.IMG_SIZE, self.IMG_SIZE, 1)
            self.data = torch.from_numpy(self.data)
        return self.data

    def __iter__(self):
        if self.alternative_data is not None:
            data = self.alternative_data
        else:
            data = self.reshaped()

        np.random.shuffle(data)
        return iter(data)