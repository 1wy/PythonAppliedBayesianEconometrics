#-*- coding:utf-8 -*-
# author:wy
# datetime:19-1-3 下午6:59
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import numpy as np
from numpy.linalg import pinv
from numpy import matmul
from scipy.io import loadmat
import pandas as pd
from MarkovSwitchingModel.utils import simS

T = 500
B1 = 0.2
B2 = 0.9
C1 = 1
C2 = -1
S1 = 3
S2 = 1
P = np.array([[0.95, 0.05],[0.05, 0.95]])
strue = np.zeros((T,2))
strue[0,0] = 1
strue = simS(strue, P)

e = np.random.randn(T,1)
Y = np.zeros((T,1))
X = np.zeros((T,1))

for i in range(1,T):
	X[i,:] = Y[i-1,:]
	if strue[i,0] == 1:
		Y[i] = np.dot([X[i,:], 1], [B1,C1]) + e[i]*np.sqrt(S1)
	else:
		Y[i] = np.dot([X[i, :], 1], [B2, C2]) + e[i] * np.sqrt(S2)

# ============================================= Run Hamilton Filter ====================================================
A = np.vstack((np.eye(2)-P, np.ones((1,2))))
EN = np.array([[0],[0],[1]])
ett11 = matmul(pinv(matmul(A.T,A)), matmul(A.T,EN))
iS1 = 1 / S1
iS2 = 1 / S2
lik = 0 # log likelihood
filter = np.zeros((T,2))

for j in range(T):
	em1 = Y[j] - np.dot([X[j,:], 1], [B1, C1])
	em2 = Y[j] - np.dot([X[j,:], 1], [B2, C2])
	neta1 = 1 / np.sqrt(S1) * np.exp(-0.5*em1*iS1*em1) # P(Y|S = 0)
	neta2 = 1 / np.sqrt(S2) * np.exp(-0.5*em2*iS2*em2) # P(Y|S = 1)

	# prediction step
	ett10 = matmul(P, ett11)
	# update step
	ett11 = ett10 * np.vstack((neta1, neta2))
	fit = np.sum(ett11) # marginal density
	ett11 /= fit
	filter[j,:] = ett11.T
	lik += np.log(fit)

print(lik)