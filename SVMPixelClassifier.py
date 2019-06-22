from Preprocessing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from itertools import product
import cv2
import numpy as np
import _pickle as pickle

TRAIN_DATA_PATH  = "./TRAIN/"
TEST_DATA_PATH = "./TEST/"


class PixelClassifier:

    def __init__(self):
        self.clf = None
        self.im_class = None
        self.im_orig = None
        self.lothres = None
        self.upthres = None
        self.im_class_thres = None
        self.filtername = None
        self.filter_d = None
        self.filter_sigmacolor = None
        self.filter_sigmaspace = None

    def trained(self):
        if self.clf is not None:
            return True
        else:
            return False

    def set_training_images(self, im_class, im_orig, lothres=np.array([0, 254, 0], dtype=np.uint8), upthres=np.array([1, 255, 1], dtype=np.uint8)):
        self.im_class = im_class
        self.im_orig = im_orig
        self.lothres = lothres
        self.upthres = upthres
        self.im_class_thres = cv2.inRange(self.im_class, self.lothres, self.upthres)
        self.im_class_thres[self.im_class_thres > 0] = 255

    def get_training_data(self, filtername, filter_d, filter_sigmacolor, filter_sigmaspace):
        if filtername == "bilateral":
            im_filter = cv2.bilateralFilter(self.im_orig, filter_d, filter_sigmacolor, filter_sigmaspace)
        else:
            raise ValueError("Only bilateral filtering is allowed.")

        train_data = []
        for labels in range(0, 5):
            for i in range(0, im_filter.shape[0]):
                for j in range(0, im_filter.shape[1]):
                    colors = tuple(im_filter[i,j])
                    # Remove purely white training data
                    if colors[0] <= 255
                    if colors[0] <= 255 or colors[1] <= 255 or colors[2] <= 255:
                    label = (self.im_class_thres[i,j],)
                    coords = (i,j)
                    train_data.append(coords + colors + label)
        train_data = np.asarray(train_data)
        print(train_data)
        return train_data

    def fit(self, n_estimators=10, filtername="bilateral", filter_d=3, filter_sigmacolor=1, filter_sigmaspace=1):
        self.filtername = filtername
        self.filter_d = filter_d
        self.filter_sigmacolor = filter_sigmacolor
        self.filter_sigmaspace = filter_sigmaspace
        train_data = self.get_training_data(filtername=filtername, filter_d=filter_d, filter_sigmacolor=filter_sigmacolor, filter_sigmaspace=filter_sigmaspace)
        # rf = RandomForestClassifier(n_estimators=n_estimators)
        # rf.fit(train_data[:,2:5], train_data[:,5])
        self.clf = rf
