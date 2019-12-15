#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:30:02 2019

Summary:
    Number of die rolls until 6 comes up and how much to bet.
    Would you bet $1 that a 6 will not be obtained within 4 rolls?
    
    Answer: No -- bc (5/6)^4 = 0.48 < 0.50.
    Would need at least K=3 for 'not obtained with K rolls'

@author: ivanbarrientos
"""

import numpy as np
from matplotlib import pyplot as plt
from random import random as unif_rand


def sample_die_from_unif(N=6):
    """Simulate an N-sided die."""
    u = unif_rand()
    x = u *  N
    for n in range(1, N+1):
        if x <= n:
            return n

def paris_salon(N=6, target=6):
    """Simulate number of N-sided die tosses until first target comes up."""
    target_reached = False
    num_tosses = 0
    while not target_reached:
        num_tosses += 1
        toss = sample_die_from_unif()
        target_reached = toss == target
    return num_tosses

n_experiments = int(1e2)  # typically want n_experiments > n_trials to better estimate final avg
# but note: variance increases as n_experiments >> n_trials
n_trials = int(1e3)  # num trials per experiment
target_after_4_trials_avg = []
avg_est_list = []
for n_exp in range(n_experiments):
    paris_trials = [paris_salon() for _ in range(n_trials)]
    target_after_4_trials_avg =  np.mean([t >= 5 for t in paris_trials])
    avg_est_list.append(np.mean(paris_trials))

#plt.hist(avg_est_list)
print(np.mean(avg_est_list))
print('target reached after 4 trials avg', np.mean(target_after_4_trials_avg))
plt.hist(paris_trials)

"""
Dice from Bernoulli
n_trials = int(1e3)
samples = []
for n in range(n_trials):
    samples.append(sample_die_from_unif())
    
counter = [0]*6
for s in samples:
    counter[s-1]+=1

freq_counter = [c/n_trials for c in counter]
print('freq counts', freq_counter)
plt.hist(samples)
"""