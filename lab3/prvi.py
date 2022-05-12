# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:06:06 2022

@author: Filip
"""

import csv

import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence

class Vocab():
    def __init__(self, frequencies, max_size=-1, min_freq=1, use_extra=True):
        self.frequencies = frequencies
        self.max_size = max_size
        self.min_freq = min_freq
        
        self.stoi = {}
        self.itos = {}
        
        temp = self.frequencies.copy()
        for key, value in temp.items():
            if value < min_freq:
                del frequencies[key]
                
        sorted_temp = [(key, value) for key, value in frequencies.items()]
        sorted_temp.sort(key=lambda x : x[1], reverse=True)
        if max_size >= 1:
            sorted_temp = sorted_temp[:max_size]
        
        frequencies = dict(sorted_temp)
        
        itos = dict()
        stoi = dict()
        
        if use_extra:        
            itos[0] = '<PAD>'
            itos[1] = '<UNK>'
            stoi['<PAD>'] = 0
            stoi['<UNK>'] = 1
        
        i = len(stoi)
        for token, freq in frequencies.items():
            itos[i] = token
            stoi[token] = i
            i += 1
        
        self.itos = itos
        self.stoi = stoi
        
    def encode(self, tokens):
        if np.isscalar(tokens):
            value = self.stoi.get(tokens, self.stoi.get('<UNK>'))
            return torch.tensor([value])
        else:
            n = len(tokens)
            values = [0] * n
            for i in range(n):
                values[i] = self.stoi.get(tokens[i], self.stoi.get('<UNK>'))
        
            return torch.tensor(values)
    
    def decode(self, values):
        if np.isscalar(values):
            return self.itos[values]
        else:
            return [self.itos[values[i]] for i in range(len(values))]
    
class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, vocab_input, vocab_labels):
        super().__init__()
        file = open(csv_path)
        instances = list()
        for line in file.readlines():
            text, label = line.replace('\n', '').split(',')
            text_list = text.split(' ')
            instances.append((text_list, label.strip()))
        
        self.instances = instances
        self.vocab_input = vocab_input
        self.vocab_labels = vocab_labels
    
    def __getitem__(self, idx):
        text, label = self.instances[idx]
        numericalized_text = self.vocab_input.encode(text)
        numericalized_label = self.vocab_labels.encode(label)
        return numericalized_text, numericalized_label
    
    def __len__(self):
        return len(self.instances)
    
        
def get_embedding_matrix(vocab, n, path=None):
    embedding_dict = {}
    if path is not None:
        file = open(path)
        for line in file.readlines():
            parts = line.replace('\n', '').split(' ')
            token = parts[0]
            values = parts[1:]
            values_vector = torch.tensor([float(x) for x in values])
            embedding_dict[token] = values_vector
        
    embedding_matrix = torch.randn((len(vocab.stoi.items()), n))
    i = 0
    for token in vocab.stoi.keys():
        if token == '<PAD>':
            embedding_matrix[i] = torch.zeros(n)
        elif token == '<UNK>':
            embedding_matrix[i] = torch.ones(n)
        elif token in embedding_dict:
            if path is not None:
                embedding_matrix[i] = embedding_dict[token]
            else:
                embedding_matrix[i] = torch.rand(n)
        i += 1
            
    return embedding_matrix    

def get_frequencies(path):
    file = open(path)
    dataset = tuple(csv.reader(file))
    freq_data = {}
    freq_label = {}
    for text, label in dataset:
        label = label.strip()
        if label not in freq_label.keys():
            freq_label[label] = 1
        else:
            freq_label[label] += 1
        
        for word in text.replace('\n', '').split(' '):
            if word not in freq_data.keys():
                freq_data[word] = 1
            else:
                freq_data[word] += 1
        
    return freq_data, freq_label

def pad_collate_fn(batch, pad_index=0):
    data, labels = zip(*batch)
    return pad_sequence(data, batch_first=True, padding_value=pad_index), torch.cat(labels, 0), torch.tensor([len(element) for element in data])
