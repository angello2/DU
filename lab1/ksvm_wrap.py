# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:45:17 2022

@author: Filip
"""

from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import data

class KSVMWrap():
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        # Konstruira omotač i uči RBF SVM klasifikator
        # X, Y_:           podatci i točni indeksi razreda
        # param_svm_c:     relativni značaj podatkovne cijene
        # param_svm_gamma: širina RBF jezgre
        self.clf = svm.SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.clf.fit(X,Y_)

    def predict(self, X):
        # Predviđa i vraća indekse razreda podataka X
        return self.clf.predict(X)

    def get_scores(self, X, Y_):
        # Vraća klasifikacijske mjere
        # (engl. classification scores) podataka X;
        # ovo će vam trebati za računanje prosječne preciznosti.
        Y = self.predict(X)
        tp = sum(np.logical_and(Y==Y_, Y_==True))
        fn = sum(np.logical_and(Y!=Y_, Y_==True))
        tn = sum(np.logical_and(Y==Y_, Y_==False))
        fp = sum(np.logical_and(Y!=Y_, Y_==False))
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp+fn + tn+fp)
        return accuracy, recall, precision

np.random.seed(100)
X, Y_ = data.sample_gmm_2d(6,2,10)
model = KSVMWrap(X,Y_)

probs = model.predict(X)
rect = (np.min(X, axis=0), np.max(X, axis=0))
data.graph_surface(lambda X: model.predict(X), rect)
data.graph_data(X, Y_, probs)
plt.show()

accuracy, recall, precision = model.get_scores(X, Y_)
print("Accuracy: ", accuracy, " Recall: ", recall, "Precision: ", precision)