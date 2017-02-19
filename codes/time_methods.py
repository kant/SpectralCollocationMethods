# -*- coding: utf-8 -*-
#Created on Oct 4, 2016
#@author: Inom Mirzaev


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from spectral_collocation import * 

import seaborn as sns

import time
from timeit import default_timer as timer

fig_params = {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"],
        'axes.facecolor':'white' , 
        'figure.facecolor':'white' }
        
sns.set( context = 'paper' , style='white', palette='deep' , font='serif' , 
        font_scale=2,  rc=fig_params)


C_g = 1

G = 1
C_a = 1.3 * G    
C_f = G  
C_mu =np.exp(-G) 



def perform_timer(myfunc, args,  n_runs=10):
    
    r_times = np.zeros(n_runs)
    
    for nn in xrange(n_runs):
        start = timer()
        
        myfunc( **args )
        end = timer()
        
        r_times[nn] = end - start    
    return np.min(r_times)
    
    
#==============================================================================
# Performance comparison of integral approximations
#==============================================================================

import simpsons_rule, gauss_rule, trapezoidal_rule, old_method

args = {'N':10, 'C_g':C_g, 'C_mu':C_mu, 'C_a':C_a , 'C_f':C_f}

dims = np.arange(10, 110, 10)

simps_time = []
gauss_time = []
trapz_time = []

for N in dims:
    args['N']=N
    simps_time.append( perform_timer( simpsons_rule.nonlinear_root , args) )
    gauss_time.append( perform_timer( gauss_rule.nonlinear_root , args) )
    trapz_time.append( perform_timer( trapezoidal_rule.nonlinear_root , args) )

    
 
plt.figure()
plt.grid(True)

plt.plot(dims, trapz_time , linewidth=1, marker='v' , markersize=10 , label='Trapezoidal')
plt.plot(dims, simps_time , linewidth=1, marker='*' , markersize=10 , label='Simpson\'s')
plt.plot(dims, gauss_time , linewidth=1, marker='o' , markersize=10 , label='Gaussian')
plt.legend()



plt.xlabel('Approximation dimension ($N$)' )
plt.ylabel( 'Best of 10 runs ($s$)' )
plt.savefig( '../images/performance_times.png' , bbox_inches='tight' , dpi=400, facecolor='white')

