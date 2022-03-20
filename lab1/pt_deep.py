# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:38:32 2022

@author: Filip
"""
import data
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

class PTDeep(nn.Module):    
    def __init__(self, parameter_list, activation=None):
        super().__init__()       
        self.cuda = torch.device('cuda')
        W=[]
        B=[]
        
        i = 0
        while i < len(parameter_list) - 1:
            w_i = nn.Parameter(torch.tensor(np.random.randn(parameter_list[i], parameter_list[i+1]), requires_grad=True, dtype=torch.float64, device=self.cuda))
            b_i = nn.Parameter(torch.zeros((1, parameter_list[i+1]), requires_grad=True, dtype=torch.float64, device=self.cuda))
            i += 1
            W.append(w_i)
            B.append(b_i)
        
        self.W = nn.ParameterList(W)
        self.B = nn.ParameterList(B)        
        self.activation_function = activation
        
    def forward(self, X):
        X = torch.tensor(X, device=self.cuda)
        i = 0
        for w, b in zip(self.W, self.B):
            X = X.mm(w) + b
            if self.activation_function is not None:
                if i < len(self.W) - 1:
                    X = self.activation_function(X)    
            i += 1
            
        probs = torch.softmax(X, dim=1).cuda()
        return probs
        
    def get_loss(self, X, Y_):
        Y = self.forward(X)
        Yoh_ = torch.tensor(data.class_to_onehot(Y_), device=self.cuda)  
        loss = -torch.log(Y) * Yoh_
        loss_sum = torch.sum(loss, dim=1)
        loss_sum_mean = torch.mean(loss_sum)
        return loss_sum_mean

    def evaluate(model, X):
        return np.argmax(model.forward(X).detach().cpu().numpy(), axis=1)
    
    def train(model, X, Y_, param_niter, param_delta):
        opt = torch.optim.SGD(params=model.parameters(), lr=param_delta)
        for i in range(param_niter):
            Y = model.forward(X)
            loss = model.get_loss(X, Y_)
            loss.backward()
                
            opt.step()
            if(i%100==0):
                print("iter: ", i, " loss: ", loss)
            opt.zero_grad()
    
    def count_parameters(self):
        sum = 0
        for parameter in self.named_parameters():
            sum += parameter[1].data.size(dim=0) * parameter[1].data.size(dim=1)
        return sum