#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:19:42 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;

def EMD2(s,t):
    cs = np.cumsum(s);
    ct = np.cumsum(t);
    return np.mean(np.power(cs-ct,2));

def dEMD2ds(s,t):
    cs = np.cumsum(s);
    ct = np.cumsum(t);
    e = cs-ct;
    r = np.cumsum(e[::-1])[::-1]/e.size;
    return r;

class learner:
    def __init__(self,X,Y):
        self.X = X;
        self.Y = Y;
        
        self.N = X.shape[1];
        self.K = Y.shape[1];
        
        
        self.V  = np.zeros(self.K);        
        self.mV = np.zeros(self.V.shape);
        self.nV = np.zeros(self.V.shape);
        
        self.B  = np.ones((self.K,self.N));
        self.mB = np.zeros(self.B.shape);
        self.nB = np.zeros(self.B.shape);
        
        self.T  = np.zeros((self.K,self.N));
        self.mT = np.zeros(self.T.shape);
        self.nT = np.zeros(self.T.shape);
               
        self.iit =  0;
        self.it  =  0;
        self.IT  = 20;
        
        self.mu  =  0.999;
        self.nu  =  0.99;
        self.eps = 1e-2;
        self.dt  = 1e-4;
                
    def eval_(self,e):
        y = np.zeros(self.K);
        x = np.zeros(self.K);
        
        for k in range(self.K):
            x[k] = self.V[k]-0.5*np.sum(np.power(self.B[k]*(e-self.T[k]),2.0));
        y = x - np.max(x);
        y = np.exp(y);
        y /= np.sum(y);
        return y;

    def get_derivs(self,e,t):
        y = np.zeros(self.K);
        x = np.zeros(self.K);
        
        for k in range(self.K):
            x[k] = self.V[k]-0.5*np.sum(np.power(self.B[k]*(e-self.T[k]),2.0));
        y = x - np.max(x);
        y = np.exp(y);
        y /= np.sum(y);
        
        dEdy = dEMD2ds(y,t);
        dEdx = np.dot(dEdy,np.diag(y)-np.outer(y,y));
        
        dEdV = dEdx;
        dEdB = np.zeros(self.B.shape);
        dEdT = np.zeros(self.T.shape);
        for k in range(self.K):
            dte = self.T[k]-e;
            dEdB[k] = -dEdx[k]*self.B[k]*np.power(dte,2);
            dEdT[k] = -dEdx[k]*np.power(self.B[k],2)*dte;
        
        return EMD2(y,t),dEdV,dEdB,dEdT;
 
    def step(self,it):
        dV = np.zeros(self.V.shape);
        dB = np.zeros(self.B.shape);
        dT = np.zeros(self.T.shape);
        
        merr = 0.0;
        p = rd.permutation(np.arange(self.X.shape[0]));
        for k in range(self.X.shape[0]):
            e = self.X[p[k]];
            t = self.Y[p[k]];
            err,a_dV,a_dB,a_dT = self.get_derivs(e,t);
            
            merr += err/self.IT;
            
            dV += a_dV/self.IT;
            dB += a_dB/self.IT;
            dT += a_dT/self.IT;
            self.it += 1;
            if self.it % self.IT == 0:
                
                #for l in range(self.L):
                #    for k in range(l):
                #        dT[l] -= (self.T[l]-self.T[k])/(1.0+0.5*np.linalg.norm(self.T[l]-self.T[k])**2);

                self.iit += 1;
                self.mV = self.mu*self.mV + (1.-self.mu)*dV;
                self.mB = self.mu*self.mB + (1.-self.mu)*dB;
                self.mT = self.mu*self.mT + (1.-self.mu)*dT;
                
                self.nV = self.nu*self.nV+(1-self.nu)*np.power(dV,2);
                self.nB = self.nu*self.nB+(1-self.nu)*np.power(dB,2);
                self.nT = self.nu*self.nT+(1-self.nu)*np.power(dT,2);
                
                mV_ = self.mV / (1.0-self.mu**(self.iit));
                nV_ = self.nV / (1.0-self.nu**(self.iit));
                mB_ = self.mB / (1.0-self.mu**(self.iit));
                nB_ = self.nB / (1.0-self.nu**(self.iit));
                mT_ = self.mT / (1.0-self.mu**(self.iit));
                nT_ = self.nT / (1.0-self.nu**(self.iit));
                
                self.V -= self.dt*mV_/(self.eps+np.sqrt(nV_));
                self.B -= self.dt*mB_/(self.eps+np.sqrt(nB_));
                self.T -= self.dt*mT_/(self.eps+np.sqrt(nT_));

                print it,self.iit, merr;                
                merr = 0.0;
                dV.fill(0.0);
                dB.fill(0.0);
                dT.fill(0.0);
