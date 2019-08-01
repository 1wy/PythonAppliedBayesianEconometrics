#-*- coding:utf-8 -*-
# author:wy
# datetime:19-1-3 下午8:36
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import itertools
import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal, gamma
from numpy import matmul, kron
from numpy.linalg import pinv, inv, cholesky, det
from numpy.random import multinomial


def log_posterior(em, b, sigma1, B0, sigma0, v0, s0, T):
	lik = -(T/2) * np.log(2*np.pi*sigma1) - 0.5 * matmul(em.T, em) / sigma1
	bprior = np.log(multivariate_normal.pdf(b, mean=B0, cov=sigma0))
	varprior = np.log(gamma.pdf(1/sigma1, v0/2, scale=2/s0))
	posterior = lik[0, 0] + bprior + varprior
	return posterior