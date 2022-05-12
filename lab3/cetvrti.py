# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:01:45 2022

@author: Filip
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:24:03 2022

@author: Filip
"""

import torch
import prvi
import numpy as np

from sklearn.metrics import confusion_matrix

class CustomRNNModel(torch.nn.Module):
    def __init__(self, embedding_matrix, text_vocab, label_vocab, net_type='RNN', hidden_size=150, num_layers=2, dropout=0., bidirectional=False):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix) 
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab
        self.params = list()
        layers = list()
        if net_type=='RNN':
            rnn1 = torch.nn.RNN(300, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional).cuda()
            rnn2 = torch.nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout).cuda()
        elif net_type=='GRU':
            rnn1 = torch.nn.GRU(300, hidden_size, num_layers, dropout=dropout).cuda()
            rnn2 = torch.nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout).cuda()
        elif net_type=='LSTM':
            rnn1 = torch.nn.LSTM(300, hidden_size, num_layers, dropout=dropout).cuda()
            rnn2 = torch.nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout).cuda()
        layers.append(rnn1)
        layers.append(rnn2)
        linear1 = torch.nn.Linear(hidden_size, hidden_size).cuda()
        layers.append(linear1)
        linear2 = torch.nn.Linear(hidden_size, 1).cuda()
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
                texts = batch[0].cuda()
                labels = batch[1].float().cuda()
                y = self.forward(texts)
                loss = self.loss(y, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            self.evaluate(valid_dataset, verbose, epoch)    
        accuracy = self.evaluate(test_dataset, verbose, 0)
        return accuracy            
            
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
                text = batch[0].cuda()
                label = batch[1].float().cuda()
                y = self.forward(text)
                y = torch.sigmoid(y).round().int()
                preds.extend(y)
                labels.extend(label)
                losses.append(self.loss(y.float(), label)) 
                
            labels = torch.tensor(labels, device = 'cpu')
            preds = torch.tensor(preds, device = 'cpu')
            confusion = confusion_matrix(labels, preds)
            tn, fp, fn, tp = confusion.ravel()
            if tp + fp != 0 and tp + fn != 0:
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
            else:
                return 0
            f1 = 2 * (precision * recall) / (precision + recall)
            loss = torch.mean(torch.tensor(losses))
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
        return accuracy
