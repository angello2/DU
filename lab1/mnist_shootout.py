# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 21:52:51 2022

@author: Filip
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

dataset_root = 'C:/Users/Filip/Documents/MNIST'  # change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

def class_to_onehot(Y):
    Yoh = np.zeros((Y.shape[0], 10))
    Yoh[range(Y.shape[0]),Y] = 1    
    return Yoh

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().add_(1).item()

print("N =",N,"D =", D,"C =", C)
    
y_train_oh = class_to_onehot(y_train)
y_test_oh = class_to_onehot(y_test)