# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:24:03 2022

@author: Filip
"""

import torch
import prvi
import numpy as np

from sklearn.metrics import confusion_matrix

class MyModel(torch.nn.Module):
    def __init__(self, embedding_matrix, text_vocab, label_vocab, dropout=0.):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix) 
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab
        self.params = list()
        layers = list()
        rnn1 = torch.nn.RNN(300, 150, 2, dropout=dropout)
        layers.append(rnn1)
        rnn2 = torch.nn.RNN(150, 150, 2, dropout=dropout)
        layers.append(rnn2)
        linear1 = torch.nn.Linear(150, 150)
        layers.append(linear1)
        linear2 = torch.nn.Linear(150, 1)
        layers.append(linear2)
        self.layers = layers
        for layer in self.layers:
            self.params.extend(layer.parameters())
        self.params.extend(self.embedding.parameters())
    def train_model(self, epochs, train_dataset, valid_dataset, test_dataset, optimizer, batch_size, grad_clip = 0., verbose=False):
        for epoch in range(1, epochs + 1):
            self.train()
            dl = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                          shuffle=True, collate_fn=prvi.pad_collate_fn)
            for batch in dl:
                texts = batch[0]
                labels = batch[1].float()
                y = self.forward(texts)
                loss = self.loss(y, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            self.evaluate(valid_dataset, verbose, epoch)    
        self.evaluate(test_dataset, verbose, 0)            
            
    def forward(self, x):
        y = self.embedding(x)
        y = torch.transpose(y, 0, 1)
        rnn1, rnn2, linear1, linear2 = self.layers[0], self.layers[1], self.layers[2], self.layers[3]
        y, h = rnn1(y, None)
        y, h = rnn2(y, h)
        y = y[-1]
        y = linear1(y)
        y = torch.relu(y)
        y = linear2(y)
        
        return y.flatten()
    def evaluate(self, dataset, verbose, epoch):
        self.eval()
        with torch.no_grad():
            dl = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, 
                                          shuffle=False, collate_fn=prvi.pad_collate_fn)
            preds = list()
            labels = list()
            losses = list()
            for batch in dl:
                text = batch[0]
                label = batch[1].float()
                y = self.forward(text)
                y = torch.sigmoid(y).round().int()
                preds.extend(y)
                labels.extend(label)
                losses.append(self.loss(y.float(), label)) 
                
            confusion = confusion_matrix(labels, preds)
            tn, fp, fn, tp = confusion.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
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
