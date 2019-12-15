#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 01:51:47 2019

Sample normal distribution by simulation of Bernoulli + CLT.

@author: ivanbarrientos
"""

import numpy as np
from matplotlib import pyplot as plt
from random import random as unif_rand

def est_sample(n_bernoulli):
    bernoulli_sum = np.sum([unif_rand() for _ in range(n_bernoulli)])
    s = (bernoulli_sum - (n_bernoulli / 2)) * (1 / (np.sqrt(n_bernoulli / 12)))
    return s

n_bernoulli = int(1e4)
n_trials = int(1e3)
sample_list = [est_sample(n_bernoulli) for _ in range(n_trials)]
plt.hist(sample_list)

# lognormal experiment
# Y is called lognoraml distr if log(Y) is normall distributed
# E(Y) = exp(mu^2 + sigma^2/2)
logn_sample_list = [np.exp(s) for s in sample_list]
plt.hist(logn_sample_list)
expected_logn_mean = np.exp(0.5)
logn_sample_mean = np.mean(logn_sample_list)
print('expected lognormal mean', expected_logn_mean)
print('simulated lognormal mean', logn_sample_mean)