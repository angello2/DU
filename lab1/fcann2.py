# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:49:06 2022

@author: Filip
"""
import data
import numpy as np
import matplotlib.pyplot as plt

class FCANN2():
    def __init__(self, D, N, C):
        # D = dimenzija ulaznih podataka
        # N = velicina skrivenog sloja
        # C = broj klasa na izlazu
        self.W1 = np.random.randn(D, N)
        self.b1 = np.zeros((1, N))
        self.W2 = np.random.randn(N, C)
        self.b2 = np.zeros((1, C))
        
        
    def forward(self, X):
        s1 = np.matmul(X, self.W1) + self.b1
        h1 = np.maximum(0., s1)
        s2 = np.matmul(h1, self.W2) + self.b2
        
        P = softmax(s2)
        return P  
    
    def get_loss(self, Y, Y_):
        # cross entropy loss
        Y_ = data.class_to_onehot(Y_)
        loss = -np.log(Y) * Y_
        loss_sum = np.sum(loss, axis=1)
        return np.mean(loss_sum)

    def train(model, X, Y_, param_niter, param_delta):
        for i in range(param_niter):
            # C = dimenzija ulaza H = velicina skrivenog sloja C = broj klasa N = broj primjera
            # forward pass -> ne koristimo funkciju forward() jer trebamo medurezultate
            s1 = np.matmul(X, model.W1) + model.b1
            s1 = X @ model.W1 + model.b1
            h1 = np.maximum(0, s1)
            s2 = np.matmul(h1, model.W2) + model.b2          
            P = softmax(s2)
            
            # Yoh_ je vektorski prikaz klase
            Yoh_ = data.class_to_onehot(Y_)
    
            # loss
            loss = model.get_loss(P, Y_)
            
            if i % 100 == 0:
                print("Iter: ", i, "loss:", loss)            
            
            # dL/dy
            dy = (P-Yoh_) / len(X)
            
            # gradijenti dL/dW2 i dL/db2
            dW2 = np.matmul(h1.T, dy)
            db2 = np.sum(dy, axis=0)
            
            # gradijent dL/ds1
            ds1 = np.matmul(dy, model.W2.T)
            ds1[h1 <= 0.0] = 0.0
            
            # gradijenti dL/dW1 i dL/db1
            dW1 = np.matmul(X.T, ds1)
            db1 = ds1.sum(axis=0)
            
            # azuriramo tezine            
            model.W1 += - param_delta * dW1
            model.b1 += - param_delta * db1
            model.W2 += - param_delta * dW2
            model.b2 += - param_delta * db2            
            
    def classify(model, X):
        return np.argmax(model.forward(X), axis=1)        

def softmax(X):
    exp = np.exp(X)
    exp_sum = np.sum(exp, axis=1)[:,np.newaxis]
    return exp / exp_sum


