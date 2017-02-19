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


N = 45



#==============================================================================
# Comparison of aggregation with fragmentation
#==============================================================================

C_g = 1
 
plt.close('all')

fig, ax = plt.subplots()


styles = [ 'solid' , 'dashed', 'dashdot', 'dotted' ]
G_vals = [0.1, 1,  10 , 100]

for nn in range(len(G_vals)):
    G = G_vals[nn]
    C_a = 1.3 * G    
    C_f = G    
    C_mu = np.exp(-G)#1 / G

    sol , x = nonlinear_root( N=N , C_g=C_g , C_mu=C_mu  , C_a = C_a , C_f = C_f )
    ax.plot( x , sol.x  , linewidth=3 , 
            linestyle = styles[ np.mod( nn , 4 ) ] ,   
            label='$\dot{\gamma}='+str(G)+'\ s^{-1}$' , color=sns.xkcd_rgb["windows blue"])
    
#ax.set_ylim(0,2)    
ax.legend( loc='upper right', bbox_to_anchor=(1, 1), frameon=True,
              fancybox=True, shadow=True,ncol=1 )
ax.set_ylabel('$p_*(x)$')
ax.set_xlabel('$x$')

fig.savefig( '../images/increasing_shear.png' , 
            bbox_inches='tight' , dpi=400, facecolor='white')    