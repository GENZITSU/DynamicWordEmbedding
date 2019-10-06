#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
utility functions for the CD method based on eq (8)
from https://arxiv.org/pdf/1703.00607.pdf
'''

import copy
import pickle

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as ss

def update(U,Y,Vm1,Vp1,lam,tau,gam,ind,embed_size, iflag):
    ''' Y : sparse matrix
    '''
    M = np.dot(U.T,U) + (lam + (1+iflag)*tau + gam)*np.eye(embed_size)
    # sprase matrix dot
    Uty = Y.dot(U).T # (r, b)
    Ub  = U[ind,:].T   # (r, b)
    A   = Uty + gam*Ub + tau*(Vm1.T+Vp1.T)  # rxb
    Vhat = np.linalg.lstsq(M,A, rcond=-1) #rxb
    return Vhat[0].T #bxr

def initvars(vocab_size,T, rank):
    '''学習対象の初期化
    '''
    U,V = [],[]
    U.append(np.random.randn(vocab_size,rank)/np.sqrt(rank))
    V.append(np.random.randn(vocab_size,rank)/np.sqrt(rank))
    for t in range(T-1):
        U.append(U[0].copy())
        V.append(V[0].copy())
    return U, V


def getmat(f, v, rowflag):
    '''ppmiのarrayからsparse matrixを作成
    '''
    with open (f, mode="rb") as f:
        data = pickle.load(f)
    # 転置
    X = ss.coo_matrix((data[:,0],(data[:,2],data[:,1])),shape=(v,v))
    if rowflag:
        X = ss.csr_matrix(X)
    else:
        X = ss.csc_matrix(X)
    return X

def getbatches(vocab,b):
    '''ボキャブラリ中からミニバッチを獲得
    '''
    batchinds = []
    current = 0
    while current<vocab:
        inds = np.arange(current,min(current+b,vocab))
        current = min(current+b,vocab)
        batchinds.append(inds)
    return batchinds

def check_diff(iteration, savefile):
    '''学習の進捗を監視する関数
    1つ前のiterationとの差分, U,Wの差分を返す
    '''
    if iteration == 0:
        return 0, 0, 0
    Ulist_old = pickle.load(open( f"{savefile}ngU_iter{iteration-1}.pickle", mode="rb" ))
    Vlist_old = pickle.load(open( f"{savefile}ngV_iter{iteration-1}.pickle", mode="rb" ))
    Ulist_new = pickle.load(open( f"{savefile}ngU_iter{iteration}.pickle", mode="rb" ))
    Vlist_new = pickle.load(open( f"{savefile}ngV_iter{iteration}.pickle", mode="rb" ))
    
    diff_U = [np.sum((new_v - old_v) ** 2, axis=1).mean() for old_v, new_v in zip(Ulist_old, Ulist_new)]
    diff_V = [np.sum((new_v - old_v) ** 2, axis=1).mean() for old_v, new_v in zip(Vlist_old, Vlist_new)]
    diff_U_V = [np.sum((U - V) ** 2, axis=1).mean() for U, V in zip(Ulist_new, Vlist_new)]
    return np.mean(diff_U), np.mean(diff_V ), np.mean(diff_U_V)
