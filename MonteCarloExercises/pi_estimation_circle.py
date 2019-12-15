#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:00:46 2019

Ch 1.
Problem 2.
Accept/reject method for estimating pi.

For a fixed number of simulations (n_sim), observe how
number of experiments (with n_trials = n_sim/n_exprmnts)
approximates the estimate of pi.

As the simulations show,:
increasing total n_experiments and decreasing n_trials, yields:
    i) better (avg) estimate of pi, but
    ii) higher variance

@author: ivanbarrientos
"""

import numpy as np
from matplotlib import pyplot as plt
from random import random as unif_rand

def circle_func(x,y):
    return x**2 + y**2

def estimate_pi(n_trials):
    """Sample x,y ~ U(0,1) and test if x^2 + y^2 < 1."""
    n_hits_in_circle = 0
    for _ in range(n_trials):
        x,y = unif_rand(), unif_rand()
        f = circle_func(x,y)
        if f <  1: 
            n_hits_in_circle += 1
    estimate = 4 * n_hits_in_circle/n_trials
    return estimate

total_simulations = int(2e4)  # total simulations 
n_experimemnts_list = [100*n for n in range(1, 101)] # block size to vary

# Execute
estimates_list = []
avg_estimates_list = []
var_estimates_list = []
for n_experiments in n_experimemnts_list:
    n_trials = int(total_simulations/n_experiments)
    for exprmt in range(n_experiments):
        est = estimate_pi(n_trials)
        estimates_list.append(est)
    # stat summaries
    avg_estimates_list.append(np.mean(estimates_list))
    var_estimates_list.append(np.var(estimates_list))
    
# plot num experiments vs estimate
plt.plot(avg_estimates_list, label='avg_est')
abs_errors_list = [np.abs(np.pi - est) for est in avg_estimates_list]
plt.plot(abs_errors_list, label='abs_err')
plt.plot(var_estimates_list, label='var_est')
plt.legend()
    

    
