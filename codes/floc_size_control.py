# -*- coding: utf-8 -*-
#Created on Oct 4, 2016
#@author: Inom Mirzaev


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from spectral_collocation import * 

import seaborn as sns


fig_params = {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"],
        'axes.facecolor':'white' , 
        'figure.facecolor':'white' }
        
sns.set( context = 'paper' , style='white', palette='deep' , font='serif' , 
        font_scale=2,  rc=fig_params)



"""
C_g = 1

C_mu = 0

ka_values = [ 1 , 10 , 100 ]

fig , ax = plt.subplots( nrows=3 , ncols=1 , sharex=True )

for pp in range(3):
    
    sol , x = nonlinear_root( 50 , C_g , C_mu, C_a = ka_values[pp] , C_f = 0.1 )

    ax[pp].plot( x , sol.x , linewidth=2)
    ax[pp].set_ylabel( '$u^*(x)$' )
    
    yticks = ax[pp].get_yticks()
    new_ticks = [np.min(yticks) ,  0.5*np.max(yticks) + 0.5*np.min(yticks) ,  np.max(yticks) ]
    ax[pp].set_yticks( new_ticks )
    
    ax[pp].text(0.5, 0.5, '$C_a='+str(ka_values[pp] ) +'$' , 
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax[pp].transAxes)
    
    if pp==2:
        ax[pp].set_xlabel('$x$')
    
fig , ax = plt.subplots( nrows=3 , ncols=1 , sharex=True )


kf_values = [0.1, 1 , 10  ]

for pp in range(3):
    
    sol , x = nonlinear_root( 50 , C_g , C_mu, C_a = 0.1 , C_f = kf_values[pp] )

    ax[pp].plot( x , sol.x , linewidth=2)
    ax[pp].set_ylabel( '$u^*(x)$' )
    
    yticks = ax[pp].get_yticks()
    new_ticks = [np.min(yticks) ,  0.5*np.max(yticks) + 0.5*np.min(yticks) ,  np.max(yticks) ]
    ax[pp].set_yticks( new_ticks )
    
    ax[pp].text(0.5, 0.5, '$C_f='+str(kf_values[pp] ) +'$' , 
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax[pp].transAxes)
    
    if pp==2:
        ax[pp].set_xlabel('$x$')
 
"""        

N = 30
C_g =0.1
C_mu = 0.01
C_a = 1
C_f = 0.01

sol , x = nonlinear_root( N=N , C_g=C_g , C_mu=C_mu  , C_a = C_a , C_f = C_f )

 

C_q = 1 / simps( renewal( x ) * sol.x , x )

sim_time = np.linspace(0, 10, 1000)

np.random.seed(seed=1)
y0 = sol.x+0.1*(0.5-np.random.rand( len(x) ) ) #10*np.ones_like( x )
#y0 = 0.1*np.ones_like( x )


yout, sim_time , x = simulate_ode( sim_time , y0, N=N , C_g=C_g , 
                                  C_mu=C_mu  , C_a = C_a , 
                                  C_f = C_f , C_q = C_q)

total_num = np.trapz( yout, x , axis=1)

fig, ax = plt.subplots()
ax.plot( sim_time , total_num )  

fig, ax = plt.subplots()
ax.plot( x , sol.x ) 
ax.plot( x , y0 ) 
ax.plot( x , yout[-1] )
#plt.savefig( '../images/agg_increasing_'+str(C_g)+'.png' , bbox_inches='tight' , dpi=400, facecolor='white')          
    