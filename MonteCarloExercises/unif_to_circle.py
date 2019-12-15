#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 00:49:04 2019

Summary:
    If x,y ~U(0,1), what is prob(x^2 +  y^2 <= 1)?
    Related to Marsaglia-Bray algo.

@author: ivanbarrientos
"""

import numpy as np
from matplotlib import pyplot as plt
from random import random as unif_rand

def estimate():
    x,y = unif_rand(), unif_rand()
    z = x**2 + y**2
    return z <= 1

n_trials = int(1e6)
samples = [estimate() for _ in range(n_trials)]
print(np.mean(samples))
print(np.pi/4)