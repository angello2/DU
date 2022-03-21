# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:38:32 2022

@author: Filip
"""
import data
import torch
import torchvision
import time
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

class PTDeep(nn.Module):    
    def __init__(self, parameter_list, activation=None):
        super().__init__()       
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')            
        self.device = device
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        torch.cuda.empty_cache()
        
        W=[]
        B=[]
        
        i = 0
        while i < len(parameter_list) - 1:
            w_i = nn.Parameter(torch.tensor(np.random.randn(parameter_list[i], parameter_list[i+1]), requires_grad=True, dtype=torch.float64, device=device))
            b_i = nn.Parameter(torch.zeros((1, parameter_list[i+1]), requires_grad=True, dtype=torch.float64, device=device))
            i += 1
            W.append(w_i)
            B.append(b_i)
        
        self.W = nn.ParameterList(W)
        self.B = nn.ParameterList(B)      
        self.activation_function = activation
        
    def forward(self, X):
        if(torch.is_tensor(X) is False):
            X = torch.tensor(X, dtype=torch.float64)
        X = X.to(self.device).double()
        i = 0
        for w, b in zip(self.W, self.B):
            X = (X.mm(w) + b)
            if self.activation_function is not None:
                if i < len(self.W) - 1:
                    X = self.activation_function(X)
            i += 1
            
        probs = torch.softmax(X, dim=1)
        return probs
        
    def get_loss(self, X, Yoh_):
        Y = self.forward(X)
        if(torch.is_tensor(Yoh_) is False):
            torch.tensor(Yoh_)
            
        loss = (-torch.log(Y) * Yoh_)
        loss_sum = torch.sum(loss, dim=1)
        loss_sum_mean = torch.mean(loss_sum)
        return loss_sum_mean

    def evaluate(model, X):
        return np.argmax(model.forward(X).detach().cpu().numpy(), axis=1)
    
    def train(model, X, Y_, param_niter, param_delta, param_lambda = 0, verbose=True, record_loss=False):
        loss_history = []
        Yoh_ = torch.tensor(data.class_to_onehot(Y_)).to(model.device)
        opt = torch.optim.SGD(params=model.parameters(), lr=param_delta)
        start_time = time.time()
        for i in range(param_niter):
            loss = model.get_loss(X, Yoh_)
            if(record_loss):
                loss_history.append(loss.detach().cpu().numpy())
            loss.backward()                
            opt.step()
            if(i%100==0):
                if(verbose):
                    print("iter: ", i, " loss: ", loss.item())
            opt.zero_grad()
        
        if(verbose):
            print("Training finished. Elapsed time: ", time.time() - start_time, " seconds. Final loss: ", loss.item())
        if(record_loss):
            return loss_history            
            
    def count_parameters(self):
        sum = 0
        for parameter in self.named_parameters():
            sum += parameter[1].data.size(dim=0) * parameter[1].data.size(dim=1)
        return sum
    