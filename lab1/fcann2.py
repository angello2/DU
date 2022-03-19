# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:49:06 2022

@author: Filip
"""
import data
import numpy as np

param_niter = 1e5
param_delta = 0.05
param_lambda = 1e-3

class FCANN2():
    def __init__(self, D, N, C):
        # D = dimenzija ulaznih podataka
        # N = velicina skrivenog sloja
        # C = broj klasa na izlazu
        self.w_0 = np.random.randn(D, N)
        self.b_0 = np.zeros((1, N))
        self.w_1 = np.random.randn(N, C)
        self.b_1 = np.zeros((1, C))
        
        
    def forward(self, X):
        # ulazni sloj
        h_1 = np.matmul(X, self.w_0) + self.b_0
        relu_1 = np.maximum(0., h_1)

        # skriveni sloj
        h_2 = np.matmul(relu_1, self.w_1) + self.b_1
        exp_2 = np.exp(h_2)
        exp_sum_2 = np.sum(exp_2, axis=1)[:,np.newaxis]
        
        # izlaz
        probs = exp_2 / exp_sum_2
        return probs  
    
    def get_loss(self, Y, Y_):
        # cross entropy loss
        Y_ = data.class_to_onehot(Y_)
        loss = -np.log(Y)
        loss_sum = np.sum(loss, axis=1)
        return np.mean(loss_sum)   

    def train(model, X, Y_, param_niter, param_delta):
        for i in range(param_niter):
            
            # forward
            h_1 = np.matmul(X, model.w_0) + model.b_0
            relu_1 = np.maximum(0., h_1)
            h_2 = np.matmul(relu_1, model.w_1) + model.b_1
            exp_2 = np.exp(h_2)
            exp_sum_2 = np.sum(exp_2, axis=1)[:,np.newaxis]
            probs = exp_2 / exp_sum_2
    
            # loss
            Yoh_ = data.class_to_onehot(Y_)
            loss = model.get_loss(probs, Y_)
            print("Iter: ", i, "loss:", loss)
            
            # gradijenti drugog sloja
            dw_1 = np.matmul(relu_1.T, (probs- Yoh_))
            db_1 = np.sum((probs - Yoh_), axis=0)
            
            
            
            dw_0 = np.matmul((probs - Yoh_), w_1)
            print("w_1:\n", model.w_1)
            print("dw_1:\n",dw_1)
            print("b_1:\n", model.b_1)
            print("db_1:\n",db_1)

model = FCANN2(2, 5, 3)
model.train([[1,1],[2,2],[3,3]],[0,1,2], 1, 0.1)