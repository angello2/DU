# -*- coding: utf-8 -*-
"""
Created on Sat May  7 21:52:17 2022

@author: Filip
"""
import torch
import prvi

import numpy as np
from sklearn.metrics import confusion_matrix

class BaselineModel(torch.nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        
        layers = list()
        layers.append(torch.nn.Linear(300, 150))
        layers.append(torch.nn.Linear(150, 150))
        layers.append(torch.nn.Linear(150, 1))
        for layer in layers:
            torch.nn.init.normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        self.layers = layers
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
    def get_params(self):
        parameters = list()
        for layer in self.layers:
            parameters.extend(layer.parameters())
        parameters.extend(self.embedding.parameters())        
        return parameters
    def forward(self, x):
        y = self.embedding(x)
        if len(y.shape) == 2:
            y = torch.mean(y, dim=0)
        else:
            y = torch.mean(y, dim=1)
        
        for layer in self.layers[:-1]:
            y = layer(y)
            y = torch.relu(y)
            
        y = self.layers[-1](y)
        return y
    def train_model(self, epochs, dataset, optimizer, batch_size, text_vocab, label_vocab, verbose=False):
        valid_dataset = prvi.NLPDataset('data/sst_valid_raw.csv', text_vocab, label_vocab)
        test_dataset= prvi.NLPDataset('data/sst_test_raw.csv', text_vocab, label_vocab)
        for epoch in range(0, epochs):   
            self.train()
            train_data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, 
                                          shuffle=True, collate_fn=prvi.pad_collate_fn)
            for batch in train_data_loader:
                optimizer.zero_grad()
                texts = batch[0]
                labels = batch[1].float()
                preds = self.forward(texts).flatten()
                loss = self.loss(preds, labels)
                loss.backward()
                optimizer.step()                
            self.evaluate(valid_dataset, epoch + 1, verbose)        
        self.evaluate(test_dataset, 0, verbose)            
    def evaluate(self, dataset, epoch, verbose=True):
        self.eval()
        with torch.no_grad():
            data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, 
                                          shuffle=False, collate_fn=prvi.pad_collate_fn)
            preds = np.zeros((len(data_loader), 1))
            labels = np.zeros((len(data_loader), 1))
            losses = list()
            for i, batch in enumerate(data_loader):
                text = batch[0]
                label = batch[1].float()
                y = self.forward(text).flatten()
                losses.append(self.loss(y, label))
                preds[i] = torch.sigmoid(y).round().int().flatten()
                labels[i] = label                
            
            confusion = confusion_matrix(labels, preds)
            tn, fp, fn, tp = confusion.ravel()
            if tp + fp != 0 and tp + fn != 0:
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
            else:
                return 0
            f1 = 2 * (precision * recall) / (precision + recall)
            loss = np.mean(losses)
            if verbose:
                if epoch == 0:
                    print('Results on test dataset:')
                else:
                    print('Epoch', epoch, 'results on validation dataset:')
                print('Accuracy:', accuracy)
                print('Precision:', precision)
                print('F1 score:', f1)            
                print('Loss:', loss)
                print('Confusion matrix:\n', confusion)
                print()
           
    