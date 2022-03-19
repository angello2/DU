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
        W=[]
        B=[]
        
        i = 0
        while i < len(parameter_list) - 1:
            w_i = nn.Parameter(torch.randn((parameter_list[i], parameter_list[i+1]), requires_grad=True, dtype=torch.float64))
            b_i = nn.Parameter(torch.zeros((1, parameter_list[i+1]), requires_grad=True, dtype=torch.float64))
            i = i + 1
            W.append(w_i)
            B.append(b_i)
        
        self.W = nn.ParameterList(W)
        self.B = nn.ParameterList(B)        
        self.activation_function = activation
        
    def forward(self, X):
        h = torch.tensor(X, dtype=torch.float64, requires_grad=False)
        for w, b in zip(self.W, self.B):
            h = h.mm(w) + b
            
            if self.activation_function is not None:
                h = self.activation_function(h)      
        return torch.softmax(h, dim=1)
        
    def get_loss(self, X, Y_):        
        loss = -torch.log(self.forward(X) + Y_)
        return torch.mean(torch.sum(loss, dim=1))

    def evaluate(model, X):
        return np.argmax(model.forward(X).detach().numpy(), axis=1)
    
    def train(model, X, Y_, param_niter, param_delta, param_lambda=0):
        opt = torch.optim.SGD(params=model.parameters(), lr=param_delta)
        
        for i in range(param_niter):
            opt.zero_grad()
            loss = model.get_loss(X, Y_)
            loss.backward()
            opt.step()
            if(i%100==0):
                print("iter: ", i, " loss: ", loss)
        
    
    def count_parameters(self):
        sum = 0
        for parameter in self.named_parameters():
            sum += parameter[1].data.size(dim=0) * parameter[1].data.size(dim=1)
        return sum


np.random.seed(100)

X,Y_ = data.sample_gmm_2d(6, 2, 10)
Yoh_ = data.class_to_onehot(Y_)

def activation(X):
    return torch.relu(X)


# definiraj model:
parameter_list = [2,10,10,2]
ptd = PTDeep(parameter_list, activation)

X = torch.from_numpy(X)
Yoh_ = torch.from_numpy(Yoh_)

ptd.train(X, Yoh_, 10000, 0.1)

# dohvati vjerojatnosti na skupu za učenje
probs = ptd.evaluate(X)

# ispiši performansu (preciznost i odziv po razredima)
accuracy, pr, M = data.eval_perf_multi(probs, Y_)
print("Accuracy: ", accuracy)
for i in range(2):
    print("Precision & recall for class", i, ": ", pr[i])


# iscrtaj rezultate, decizijsku plohu
rect = (np.min(X.numpy(), axis=0), np.max(X.numpy(), axis=0))
data.graph_surface(lambda X: ptd.evaluate(X), rect)
data.graph_data(X, Y_, probs)
plt.show()