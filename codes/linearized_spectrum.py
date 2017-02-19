# -*- coding: utf-8 -*-
#Created on Oct 4, 2016
#@author: Inom Mirzaev

from __future__ import division

import gauss_rule as gr
import matplotlib.pyplot as plt
from matplotlib import ticker
    

import time
import numpy as np
start = time.time()

C_g = 1

G = 10
C_a = 1.3 * G    
C_f = G  
C_mu =np.exp(-G) 

import seaborn as sns


fig_params = {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"],
        'axes.facecolor':'white' , 
        'figure.facecolor':'white' }
        
sns.set( context = 'paper' , style='white', palette='deep' , font='serif' , 
        font_scale=2,  rc=fig_params)


#==============================================================================
#   Convergence of eigenvalues
#==============================================================================
lead_eigs = []

dims = np.arange(10, 160, 10)

for N in dims:
    sol, x, eigs = gr.linearized_spectrum( N , C_g , C_mu, C_a=C_a, C_f=C_f )
    lead_eigs.append( np.max( np.real(eigs) ) )
   
plt.figure()

plt.plot( dims , lead_eigs , linewidth=1, marker='o', markersize=10)

plt.savefig( '../images/evalue_convergence.png'  , dpi=400, facecolor='white')



#==============================================================================
# Example Spectrum
#==============================================================================
N=50

fig, ax = plt.subplots()

sol, x, eigs = gr.linearized_spectrum( N , C_g , C_mu, C_a=C_a, C_f=C_f )
ax.scatter(np.real(eigs) , np.imag(eigs) , s=20)

ax.grid(True, which='both')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.yaxis.tick_left()
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.xaxis.tick_bottom()
fig.savefig( '../images/example_spectrum_50.png'  , dpi=400, facecolor='white')


N=100

fig, ax = plt.subplots()

sol, x, eigs = gr.linearized_spectrum( N , C_g , C_mu, C_a=C_a, C_f=C_f )
ax.scatter(np.real(eigs) , np.imag(eigs) , s=20)

ax.grid(True, which='both')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.yaxis.tick_left()
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.xaxis.tick_bottom()

fig.savefig( '../images/example_spectrum_100.png'  , dpi=400, facecolor='white')


#==============================================================================
# Leading eigenvalues for various growth and shear rates
#==============================================================================


N=50

Cg_values = np.linspace(0.01, 1, 40 )

G_values = np.linspace( 0.01 , 50, 40 )

leading_evals  = np.zeros( [ len(Cg_values) , len(G_values) ] )

for mm in xrange( len( Cg_values ) ):
    for nn in xrange( len( G_values ) ):
        
        C_g = Cg_values[mm]
        
        G = G_values[nn]
        C_a = 1.3 * G    
        C_f = G  
        C_mu =np.exp(-G) 
        
        sol, x, eigs = gr.linearized_spectrum( N , C_g , C_mu, C_a=C_a, C_f=C_f )
        leading_evals[ mm , nn ] = np.max( np.real(eigs) )      
        

fig, ax = plt.subplots()
ax.patch.set_facecolor('#FFFACD')
im  = ax.imshow( leading_evals , cmap='Dark2' , vmax=0,  
                origin='lower' ,  interpolation='gaussian' )

ax.set_yticks( np.linspace(0, len(Cg_values)-1 , 6 , endpoint=True) )
ax.set_yticklabels( np.round( np.linspace(np.min(Cg_values) , np.max(Cg_values) ,  6) , 2) )


ax.set_xticks( np.linspace(0, len(G_values)-1 , 6, endpoint=True) )
ax.set_xticklabels( np.int_( np.linspace(np.min(G_values) , np.max(G_values) ,  6) ) , rotation=90)


ax.set_ylabel('$C_g$')
ax.set_xlabel( '$\dot{\gamma}$' )
ax.tick_params(labelbottom='on',labeltop='off')

cb = fig.colorbar( im ,  fraction=0.046, pad=0.04 )    

tick_locator = ticker.MaxNLocator(nbins=6)
cb.locator = tick_locator
cb.update_ticks()
        
fig.savefig( '../images/evals_shear_growth.png' , 
            bbox_inches='tight' , dpi=400, facecolor='white')       



end = time.time()

print "Elapsed time", round( end - start   , 2 ) ,  "seconds "


