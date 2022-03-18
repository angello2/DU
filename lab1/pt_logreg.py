# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:50:25 2022

@author: Filip
"""

import data
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch import nn

class PTLogreg(nn.Module):    
    def __init__(self, D, C):
        """Arguments:
            - D: dimensions of each datapoint 
            - C: number of classes
        """    
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        super().__init__()
        self.W = nn.Parameter(torch.randn((D, C)), requires_grad=True)
        self.b = nn.Parameter(torch.randn(C), requires_grad=True)

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        # ...
        logits = X.mm(self.W) + self.b
        probs = torch.softmax(logits, dim=1)
        return probs
    

    def get_loss(self, X, Y_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        # ...
        Y = self.forward(X)
        
        log_loss = torch.log(Y + 1e-13) * Y_ # dodajemo jako mali broj da izbjegnemo log(0)
        loss_sum = torch.sum(log_loss, dim=1)
        loss_mean = torch.mean(loss_sum)
        
        return -1 * loss_mean        

    def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0):
        """Arguments:
            - X: model inputs [NxD], type: torch.Tensor
            - Yoh_: ground truth [NxC], type: torch.Tensor
            - param_niter: number of training iterations
            - param_delta: learning rate
            - param_lambda: regularization factor
        """
  
        # inicijalizacija optimizatora
        # ...
        opt = torch.optim.SGD(params=model.parameters(), lr = param_delta)
        
        # petlja učenja
        # ispisujte gubitak tijekom učenja
        # ...
        for i in range(param_niter):
            loss = model.get_loss(X,Yoh_) + param_lambda * torch.norm(model.W)
            loss.backward()
            opt.step()
            if(i % 100 == 0):
                print(f"iter: {i} loss: {loss:.06f}")
                
            opt.zero_grad()

    def evaluate(model, X):
        """Arguments:
            - model: type: PTLogreg
            - X: actual datapoints [NxD], type: np.array
            Returns: predicted class probabilites [NxC], type: np.array
        """
        # ulaz je potrebno pretvoriti u torch.Tensor
        # izlaze je potrebno pretvoriti u numpy.array
        # koristite torch.Tensor.detach() i torch.Tensor.numpy()
        X = torch.tensor(X).float()
        Y = model.forward(X).detach().numpy()
        return np.argmax(Y, axis=1)
    