#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:49:40 2019

@author: ivanbarrientos
"""

import numpy as np
import itertools
from matplotlib import pyplot as plt
from random import random as unif_rand

def sample_pt(d=2):
    """Sample a d-dimensional point in the cube."""
    pt = [unif_rand() for _ in range(d)]
    return pt

def generate_pts(N, d=2):
    """Generate N-many d-dimensional pts in cube."""
    samples = [sample_pt(d) for _ in range(N)]
    return samples

def compute_min_dist(samples):
    #for a, b in itertools.product(samples, samples)
    min_dist = np.inf
    for a, b in itertools.combinations(X,2):
        diff = np.array(a) - np.array(b)
        dist = np.linalg.norm(diff, ord=2)
        if dist < min_dist:
            min_dist = dist
    
    return min_dist

dim = 2
n_exp = int(1e2)
n_trials_per_exp = int(1e2)
n_samples_list = range(2,11)
N_avg_list = []
for N in n_samples_list:
    print(N)
    min_dist_averages = []
    for e in range(n_exp):
        min_dist_experiment_list = []
        for trial in range(n_trials_per_exp):
            X = generate_pts(N=N, d=dim)
            min_dist_trial = compute_min_dist(X)
            min_dist_experiment_list.append(min_dist_trial)
        experiment_avg = np.mean(min_dist_experiment_list)
        min_dist_averages.append(experiment_avg)
    N_avg = np.mean(min_dist_averages)
    N_avg_list.append(N_avg)

# Summarize
plt.plot(N_avg_list)
#plt.hist(min_dist_averages)
"""
Z = np.reshape(X, (n_samples, dim))
min_dist = compute_min_dist(Z)
print(min_dist)
if dim == 2:
    x = [z[0] for z in Z]
    y = [z[1] for z in Z]
    plt.scatter(x,y)
"""
    
