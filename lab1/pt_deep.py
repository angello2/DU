# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:38:32 2022

@author: Filip
"""
import torch
import numpy as np
from torch import nn

class PTDeep(nn.Module):
    def __init__(self, D, C):
        