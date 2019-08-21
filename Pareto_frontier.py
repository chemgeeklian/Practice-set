#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 18:30:16 2019
https://pythonhealthcare.org/tag/pareto-front/

Slightly modify the codes to make it able to run in numba for faster computation.

@author: xinran
"""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt

mu, sigma = 0, 0.1 # mean and standard deviation
np.random.seed(23)
s = np.random.normal(mu, sigma, [600000,2])

@jit(nopython = True)
def identify_pareto(scores):
    # Must input scores as np.ndarray
    population_size = scores.shape[0]
    
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size)

    for i in range(population_size):
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
            
    # Return ids of scenarios on pareto front
    return population_ids[np.where(pareto_front==1)]

x = s[:, 0]
y = s[:, 1]

a = identify_pareto(s)

# Scatter the points with the ones on the pareto frontier.
plt.figure(figsize = [5,5])
plt.scatter(x,y,s=10)
plt.scatter(x[a],y[a])
plt.xlabel('Objective A')
plt.ylabel('Objective B')
plt.show()
