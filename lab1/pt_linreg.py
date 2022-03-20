# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:27:22 2022

@author: Filip
"""

import torch
import torch.nn as nn
import torch.optim as optim



## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2])
Y = torch.tensor([3, 5])

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.1)

for i in range(100):
    # afin regresijski model
    Y_ = a * X + b

    diff = (Y-Y_)

    # srednji kvadratni gubitak - osiguran rad neovisno o broju ulaznih tocaka
    loss = torch.mean(diff**2)

    # računanje gradijenata
    loss.backward()

    real_grad_a = 2 * torch.mean(-diff * X) 
    real_grad_b = 2 * torch.mean(-diff)
    print(f'step: {i:2d}, loss: {loss:.6f}, PyTorch grad: a={a.grad.detach().numpy()[0]:.03f} b={b.grad.detach().numpy()[0]:.03f}, Analitički grad: a={real_grad_a:.03f} b={real_grad_b:.03f}')
    
    # korak optimizacije
    optimizer.step()

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()