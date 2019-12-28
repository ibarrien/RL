#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:18:17 2019

From Ch1
Problem 9

@author: ivanbarrientos
"""

import time
import numpy as np
from matplotlib import pyplot as plt
from random import random as unif_rand

def simulate_gambler(max_iter=1e6, gambler_limit=100, house_limit=200,
                     gambler_win_prob=0.5):
    """Gambler wins with prob > 0 and the game stops when either:
        i) the gambler runs out of money, or
        ii) house runs out of money, or
        iii) max_iter reached.
    """
    curr_iter = 0
    gambler_val_path = [gambler_limit]  # cumulative win/loss each round
    while gambler_limit and house_limit and curr_iter < max_iter:
        curr_iter += 1
        u = unif_rand()
        payout = 1 if u > gambler_win_prob else -1
        gambler_limit += payout
        house_limit += payout
        gambler_val_path += [gambler_limit]
    return gambler_val_path, curr_iter

# estimate avg and variance of duration as function of max iter
# interesting problem: what max values should iter and n_trials be?
start_time = time.time()
max_range = 5
scale_iter = 1e5
max_iter_list = [n*scale_iter for n in range(1, max_range)]
n_trials = 250
avg_duration_list = []
var_duration_list = []
avg_max_iter_reached_list = []
for max_iter in max_iter_list:
    duration_list = []
    max_iter_reached_list = []
    print(int(max_iter/scale_iter))
    for _ in range(n_trials):
        sample_path, n_iter = simulate_gambler(max_iter=max_iter)
        duration_list.append(n_iter)
        max_iter_reached_list.append(n_iter==max_iter)
    # compute stats based on this fixed max_iter
    avg_duration_list.append(np.mean(duration_list))
    var_duration_list.append(np.var(duration_list))
    # frequency of times max_iter reached (gambler/house does not win)
    avg_max_iter_reached_list.append(np.mean(max_iter_reached_list))

end_time = time.time()
work_time = end_time -  start_time
print(work_time)
# PLOT
plt.plot(avg_duration_list)
plt.plot(var_duration_list)
#plt.plot(sample_path)
#print(n_iter)
