import torch as t
import cv2
import numpy as np



def preprocess(img):
    pass
    return img

class Transform(object):
    def __init__(self):
        pass
    def __call__(self, in_data):
        img, label = in_data
        img = preprocess(img, self.min_size, self.max_size)

        return img, label

class Dataset:
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass
        return img.copy(), label.copy()

    def __len__(self):
        pass

