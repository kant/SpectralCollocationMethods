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

C_mu = 0.01

#==============================================================================
# Comparison of aggregation with fragmentation
#==============================================================================

C_g = 1

Ca_values = [0.01 , 0.1, 1 , 2 , 6, 10]#np.linspace(0, 100, 10 )

Cf_values = [0.01 , 0.1, 1 , 2 , 6, 10]#np.linspace( 0 , 10, 10)

biomass  = np.zeros( [ len(Ca_values) , len(Cf_values) ] )

for mm in xrange( len( Ca_values ) ):
    for nn in xrange( len( Cf_values ) ):
        sol , x = nonlinear_root( N=N , C_g=C_g , C_mu=C_mu  , C_a = Ca_values[mm] , C_f = Cf_values[nn] )
        biomass[ mm , nn ] = simps( x*sol.x , x)        
        
 
plt.close('all')
fig, ax = plt.subplots()

im  = ax.matshow( biomass , cmap='Blues' , vmin=0 ,    origin='lower' )
ax.set_yticklabels( [0]+Ca_values )
ax.set_ylabel('$C_a$')

ax.set_xticklabels(  [0]+Cf_values , rotation=30)

ax.set_xlabel( '$C_f$' )
ax.tick_params(labelbottom='on',labeltop='off')

cb = fig.colorbar( im ,  fraction=0.046, pad=0.04 )    

from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=6)
cb.locator = tick_locator
cb.update_ticks()
        
fig.savefig( '../images/agg_frag_effect.png' , bbox_inches='tight' , dpi=400, facecolor='white')          

#==============================================================================
# Comparison of growth with fragmentation
#==============================================================================
C_a = 1


Cg_values = [0.01 , 0.1, 1 , 2 , 6, 10]#[0.001 , 0.01, 0.1 , 0.5 , 1, 2]#np.linspace(0, 100, 10 )

Cf_values = [0.01 , 0.1, 1 , 2 , 6, 10]#[0.01 , 0.1, 1,  5, 10 , 20]#np.linspace( 0 , 10, 10)

biomass  = np.zeros( [ len(Cg_values) , len(Cf_values) ] )

for mm in xrange( len( Cg_values ) ):
    for nn in xrange( len( Cf_values ) ):
        sol , x = nonlinear_root( N=N , C_a=C_a , C_mu=C_mu  , C_g = Cg_values[mm] , C_f = Cf_values[nn] )
        biomass[ mm , nn ] = simps( x*sol.x , x)       
        

fig, ax = plt.subplots()

im  = ax.matshow( biomass , cmap='Blues' , vmin=0 ,  origin='lower' )
ax.set_yticklabels( [0]+Cg_values )
ax.set_ylabel('$C_g$')

ax.set_xticklabels(  [0]+Cf_values , rotation=30)

ax.set_xlabel( '$C_f$' )
ax.tick_params(labelbottom='on',labeltop='off')

cb = fig.colorbar( im ,  fraction=0.046, pad=0.04 )    

from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=6)
cb.locator = tick_locator
cb.update_ticks()
        
fig.savefig( '../images/growth_frag_effect.png' , 
            bbox_inches='tight' , dpi=400, facecolor='white')          


#==============================================================================
# Comparison of growth with aggregation
#==============================================================================
C_f = 1

Cg_values = [0.01 , 0.1, 1 , 2 , 6, 10]#[0.001 , 0.01, 0.1 , 0.3 , 0.6, 1]#np.linspace(0, 100, 10 )

Ca_values = [0.01 , 0.1, 1 , 2 , 6, 10]#[0.001 , 0.01, 0.1 , 0.5 , 1, 2]

biomass  = np.zeros( [ len(Cg_values) , len(Ca_values) ] )

for mm in xrange( len( Cg_values ) ):
    for nn in xrange( len( Ca_values ) ):
        sol , x = nonlinear_root( N=N , C_f=C_f , C_mu=C_mu  , C_g = Cg_values[mm] , C_a = Ca_values[nn] )
        biomass[ mm , nn ] = simps( x*sol.x , x)        
        

fig, ax = plt.subplots()

im  = ax.matshow( biomass , cmap='Blues' , vmin=0 ,   origin='lower' )
ax.set_yticklabels( [0]+Cg_values )
ax.set_ylabel('$C_g$')

ax.set_xticklabels(  [0]+Ca_values , rotation=30)

ax.set_xlabel( '$C_a$' )
ax.tick_params(labelbottom='on',labeltop='off')

cb = fig.colorbar( im ,  fraction=0.046, pad=0.04 )    

from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=6)
cb.locator = tick_locator
cb.update_ticks()
        
fig.savefig( '../images/growth_agg_effect.png' , 
            bbox_inches='tight' , dpi=400, facecolor='white')          
