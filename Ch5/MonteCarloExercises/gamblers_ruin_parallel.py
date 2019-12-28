#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:04:30 2019

@author: ivanbarrientos
"""

import time
import numpy as np
from multiprocessing import Pool
from matplotlib import pyplot as plt
from random import random as unif_rand

def simulate_game(max_iter):
    """Gambler wins with prob > 0 and the game stops when either:
        i) the gambler runs out of money, or
        ii) house runs out of money, or
        iii) max_iter reached.
    
    Params:
        max_iter (int): upper bound on number of rounds in game.
    
    Returns:
        curr_iter (int): num iterations reached at end of game.
    
    Notes:
        "game" is synonymous with "trial".
    """
    # hardcoded vals to make parallelization easier...
    gambler_limit=100
    house_limit=200
    gambler_win_prob=0.5
    curr_iter = 0
    # simulate game
    while gambler_limit and house_limit and curr_iter < max_iter:
        curr_iter += 1
        u = unif_rand()
        payout = 1 if u > gambler_win_prob else -1
        gambler_limit += payout
        house_limit += -payout
    return curr_iter

class GamblerStats(object):
    """Compute and record stats over several experiments."""
    
    def __init__(self, avg_duration_list=[], var_duration_list=[], 
                 avg_max_iter_reached_list=[]):
        self.avg_duration_list = avg_duration_list
        self.var_duration_list = var_duration_list
        self.avg_max_iter_reached_list = avg_max_iter_reached_list
        
    def compute_duration_stats(self, duration_list, max_iter):
        # first and second order moments
        self.avg_duration = np.mean(duration_list)
        self.var_duration = np.var(duration_list)
        # frequency of max iter reached
        self.max_iter_reached_list = [d == max_iter for d in duration_list]
        self.avg_max_iter_reached = np.mean(self.max_iter_reached_list)
        return None
        
    def update_stats(self, duration_list, max_iter):
        """Helper method to update stats based on new duration_list."""
        self.compute_duration_stats(duration_list, max_iter)
        self.avg_duration_list.append(self.avg_duration)
        self.var_duration_list.append(self.var_duration)
        self.avg_max_iter_reached_list.append(self.avg_max_iter_reached)
        return None
        
if __name__ == "__main__":
    start_time = time.time()
    procs = 3  # num physical cores - 1
    max_range = 20
    scale_iter = 2e4
    max_iter_list = [n*scale_iter for n in range(1, max_range)]  # max iter per trial
    n_trials = 250  # num trials per experiment
    # Execute Simulated Experiments
    gambler_stats = GamblerStats()
    for max_iter in max_iter_list:
        # parallel compute trials
        jobs = [max_iter] * n_trials
        duration_list = Pool(procs).map(simulate_game, jobs)
        # Update Gambler Stats
        gambler_stats.update_stats(duration_list, max_iter)
    # Summarize Time
    end_time = time.time()
    work_time = end_time - start_time
    print('Time taken for all simulations', work_time)
    # Example Plot
    plt.title('Avg Game Duration [iter]')
    plt.xlabel('max_iter x {:.0E}'.format(scale_iter))
    plt.ylabel('Avg Duration')
    plt.plot(gambler_stats.avg_duration_list, '.-')
    print(max(gambler_stats.avg_max_iter_reached_list))
    """

    plt.plot(avg_duration_list)
    plt.plot(var_duration_list)
    """