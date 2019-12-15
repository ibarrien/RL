#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:57:03 2019

@author: ivanbarrientos
"""

import time
import numpy as np
#import pandas as pd
from matplotlib import pyplot as plt
from random import random as unif_rand

def play_game():
    player_amt = -5  # player initially pays 5 units
    u = unif_rand()
    success_counter = 0
    payout = 1
    while u > 0.5:
        success_counter += 1
        payout = 2*payout
        player_amt += payout
        u = unif_rand()
    return player_amt
   
    
def simulate_trials(n_trials):
    player_amt_list = [play_game() for _ in range(n_trials)]
    return player_amt_list

n_trials_list = [10**n for n in range(1, 8)]
avg_payout_list = []
for n_trials in n_trials_list:
    player_amt_list = simulate_trials(n_trials)
    #plt.hist(player_amt_list)
    avg_payout_list.append(np.mean(player_amt_list))
    
plt.plot(avg_payout_list)