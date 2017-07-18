#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:03:51 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;

import scipy.io as scio;
import matplotlib.pyplot as plt;
import matplotlib.image  as img;

from emd2_loss import *;

a = scio.loadmat("mnist_all.mat");

D = 10;
I = 50;
E_train = np.zeros((D*I,28*28));
T_train = 1e-8*np.ones((D*I,10));

E_test = np.zeros((2*D*I,28*28));
T_test = 1e-8*np.ones((2*D*I,10));
for i in range(I):
    for b in range(D):
        E_train[D*i+b]    = a['train'+str(b)][i,:]/255.0;
        T_train[D*i+b][b] = 1.0;
        T_train[D*i+b] /= np.sum(T_train[D*i+b]);

for i in range(2*I):
    for b in range(D):
        E_test[D*i+b]    = a['test'+str(b)][i,:]/255.0;
        T_test[D*i+b][b] = 1.0;
        T_test[D*i+b] /= np.sum(T_test[D*i+b]);

net = learner(E_train,T_train);

it = 0;
net.iit = 0;
while it < 100:
    net.step(it);
    mclass = 0.0;
    merr   = 0.0;
    for t in range(E_test.shape[0]):
        z = net.eval_(E_test[t]);
        mclass += (np.argmax(z) == np.argmax(T_test[t]));
        merr   += EMD2(z,T_test[t]);
    print "-->",it, mclass*100./E_test.shape[0], merr/E_test.shape[0];
    it += 1;

f,axarr = plt.subplots(2,D);
for l in range(D):
    
    axarr[0,l].imshow(net.B[l].reshape((28,28)),cmap='gray');
    axarr[0,l].set_xticklabels([]);
    axarr[0,l].set_yticklabels([]);
    axarr[0,l].grid(False);
    
    axarr[1,l].imshow(net.T[l].reshape((28,28)),cmap='gray');
    axarr[1,l].set_xticklabels([]);
    axarr[1,l].set_yticklabels([]);
    axarr[1,l].grid(False);
plt.show()
