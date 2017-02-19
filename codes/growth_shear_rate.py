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


N = 25


#==============================================================================
# Average floc size for different growth and shear rates
#==============================================================================

Cg_values = 0.5*2.0**( np.arange(5) )#[ 0.5 , 1 , 2 , 4 ]

G_vals = [0.01, 0.1, 1, 10, 50, 100] #+ list( 10*2.0**( np.arange(5) ) )#[1, 10 , 100, 500]

avg_floc_size  = np.zeros( [ len(Cg_values) , len(G_vals) ] )

Cq_values  = np.zeros( [ len(Cg_values) , len(G_vals) ] )

for mm in xrange( len( Cg_values ) ):
    for nn in xrange( len( G_vals ) ):
        
        G = G_vals[nn]
        C_a = 1.3 * G    
        C_f = G
        C_mu = np.exp(-G)
        
        sol , x = nonlinear_root( N=N , C_f=C_f , C_mu=C_mu  , C_g = Cg_values[mm] , C_a = C_a )
        cum_dist = np.cumsum( sol.x )
        avg_idx = np.abs( cum_dist - cum_dist[-1]/2 ).argmin()
        
        avg_floc_size[ mm , nn ] = x[ avg_idx ]        
        
        Cq_values[ mm , nn ] = 1 / simps( renewal(x)*sol.x , x)        
        

plt.close('all')
fig, ax = plt.subplots()

im  = ax.matshow( avg_floc_size , cmap='Blues' , vmin=0 ,   origin='lower' )
ax.set_yticklabels( [0]+list(Cg_values) )
ax.set_ylabel('$C_g$')

ax.set_xticklabels(  [0]+list( G_vals ) , rotation=30)

ax.set_xlabel( '$\dot{\gamma}$' )
ax.tick_params( labelbottom='on' , labeltop='off' )

cb = fig.colorbar( im ,  fraction=0.046, pad=0.04 )    

from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=6)
cb.locator = tick_locator
cb.update_ticks()
cb.set_label('Average floc size', rotation=90 , labelpad=3)
        
fig.savefig( '../images/growth_shear.png' , 
            bbox_inches='tight' , dpi=400, facecolor='white')          


#==============================================================================
# Renewal rate
#==============================================================================

fig, ax = plt.subplots()

im  = ax.matshow( Cq_values , cmap='Blues' , vmin=0 ,   origin='lower' )
ax.set_yticklabels( [0]+list(Cg_values) )
ax.set_ylabel('$C_g$')

ax.set_xticklabels(  [0]+list( G_vals ) , rotation=30)

ax.set_xlabel( '$\dot{\gamma}$' )
ax.tick_params( labelbottom='on' , labeltop='off' )

cb = fig.colorbar( im ,  fraction=0.046, pad=0.04 )    

from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=6)
cb.locator = tick_locator
cb.update_ticks()
cb.set_label('Renewal rate', rotation=90 , labelpad=3)
        
fig.savefig( '../images/renewal_shear.png' , 
            bbox_inches='tight' , dpi=400, facecolor='white')    

     

