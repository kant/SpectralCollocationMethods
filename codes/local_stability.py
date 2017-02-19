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



N = 40

C_g = 1

G = 10
C_a = 1.3 * G    
C_f = G  
C_mu = np.exp(-G)

sol , x = nonlinear_root( N=N , C_f=C_f , C_mu=C_mu  , 
                                 C_g = C_g , C_a = C_a )
        
C_q = 1 / simps( renewal(x)*sol.x , x)  


sim_time = np.linspace(0, 2, 500)


        
plt.close('all')

fig, ax = plt.subplots( nrows=2 , ncols=1 , sharex=True  )
fig1, ax1 = plt.subplots(  )
ax1.plot( x , sol.x , linewidth=2, color='k')


#==============================================================================
# Initial condition #1
#==============================================================================
np.random.seed(0)
#dummy = np.random.uniform( np.min( sol.x ) , np.max( sol.x ) )
dummy  = 1
y0 = 0.5*dummy*np.random.uniform( -1, 1 , len(x) ) + dummy*np.ones_like(x)

#y0 = np.min( sol.x )*np.random.uniform( -1, 1 , len(x) ) + sol.x

yout, sim_time , x = simulate_ode( sim_time , y0 ,  C_g=C_g , 
                                   C_mu=C_mu , C_q = C_q , 
                                   C_a = C_a , C_f=C_f , N=N )        


ax[0].plot( sim_time , simps( yout , x , axis=1) , linewidth=2 , linestyle='solid' )
ax[1].plot( sim_time , simps( x*yout , x , axis=1) , linewidth=2 , linestyle='solid')
ax1.plot( x , y0 , linewidth=2 , linestyle='solid' )

#==============================================================================
# Initial condition #2
#==============================================================================
np.random.seed(1)

#y0 = x**(1/3) + np.min(sol.x)*np.random.uniform( 0, 1 , len(x) )
y0  = sol.x + sol.x*np.random.uniform( 0, 1 , len(x) )
yout, sim_time , x = simulate_ode( sim_time , y0 ,  C_g=C_g , 
                                   C_mu=C_mu , C_q = C_q , 
                                   C_a = C_a , C_f=C_f , N=N )        


ax[0].plot( sim_time , simps( yout , x , axis=1) , linewidth=2 , linestyle='solid' )
ax[1].plot( sim_time , simps( x*yout , x , axis=1) , linewidth=2, linestyle='solid' )
ax1.plot( x , y0 , linewidth=2 , linestyle='solid')
#==============================================================================
# Initial condition #3
#==============================================================================
np.random.seed(110)

dummy  = np.min( sol.x )
y0 = 0.5*dummy*np.random.uniform( -1, 1 , len(x) ) + dummy*np.ones_like(x)
yout, sim_time , x = simulate_ode( sim_time , y0 ,  C_g=C_g , 
                                   C_mu=C_mu , C_q = C_q , 
                                   C_a = C_a , C_f=C_f , N=N )        


ax[0].plot( sim_time , simps( yout , x , axis=1) , linewidth=2 , linestyle='solid')
ax[1].plot( sim_time , simps( x*yout , x , axis=1) , linewidth=2, linestyle='solid' )
ax1.plot( x , y0 , linewidth=2 , linestyle='solid')


#==============================================================================
# Final touches
#==============================================================================
ax[1].axhline(y=simps( x*sol.x , x) , linewidth=3 , color = 'k')
ax[1].locator_params(axis='y',nbins=4)
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$M_1(t)$')

ax[0].axhline(y=simps( sol.x , x) , linewidth=3 , color = 'k')
ax[0].locator_params(axis='y',nbins=4)
ax[0].set_ylabel('$M_0(t)$')

ax1.set_ylabel('$u_*(x)$')
ax1.set_xlabel('$x$')
        
fig.savefig( '../images/stability_moments.png' , bbox_inches='tight' , dpi=400, facecolor='white')              
fig1.savefig( '../images/stability_initial.png' , bbox_inches='tight' , dpi=400, facecolor='white')          
  



#==============================================================================
# Plots for Instability
#==============================================================================
N = 40

C_g = 0.1

G = 10
C_a = 1.3 * G    
C_f = G  
C_mu = np.exp(-G)

sol , x = nonlinear_root( N=N , C_f=C_f , C_mu=C_mu  , 
                                 C_g = C_g , C_a = C_a )
        
C_q = 1 / simps( renewal(x)*sol.x , x)  


sim_time = np.linspace(0, 2, 500)


fig, ax = plt.subplots( nrows=2 , ncols=1 , sharex=True  )
fig1, ax1 = plt.subplots(  )
ax1.plot( x , sol.x , linewidth=2, color='k')


#==============================================================================
# Initial condition #1
#==============================================================================
np.random.seed(0)
#dummy = np.random.uniform( np.min( sol.x ) , np.max( sol.x ) )
dummy  = 1.5
y0 = dummy*np.random.uniform( -1, 1 , len(x) ) + dummy*np.ones_like(x)

#y0 = np.min( sol.x )*np.random.uniform( -1, 1 , len(x) ) + sol.x

yout, sim_time , x = simulate_ode( sim_time , y0 ,  C_g=C_g , 
                                   C_mu=C_mu , C_q = C_q , 
                                   C_a = C_a , C_f=C_f , N=N )        


ax[0].plot( sim_time , simps( yout , x , axis=1) , linewidth=2 , linestyle='solid' )
ax[1].plot( sim_time , simps( x*yout , x , axis=1) , linewidth=2 , linestyle='solid')
ax1.plot( x , y0 , linewidth=2 , linestyle='solid' )

#==============================================================================
# Initial condition #2
#==============================================================================
np.random.seed(1)

#y0 = x**(1/3) + np.min(sol.x)*np.random.uniform( 0, 1 , len(x) )
y0  = sol.x + sol.x*np.random.uniform( 0, 1 , len(x) )
yout, sim_time , x = simulate_ode( sim_time , y0 ,  C_g=C_g , 
                                   C_mu=C_mu , C_q = C_q , 
                                   C_a = C_a , C_f=C_f , N=N )        


ax[0].plot( sim_time , simps( yout , x , axis=1) , linewidth=2 , linestyle='solid' )
ax[1].plot( sim_time , simps( x*yout , x , axis=1) , linewidth=2, linestyle='solid' )
ax1.plot( x , y0 , linewidth=2 , linestyle='solid')
#==============================================================================
# Initial condition #3
#==============================================================================
np.random.seed(110)

dummy  = 1#np.min( sol.x )
y0 = 0.5*dummy*np.random.uniform( -1, 1 , len(x) ) + dummy*np.ones_like(x)
yout, sim_time , x = simulate_ode( sim_time , y0 ,  C_g=C_g , 
                                   C_mu=C_mu , C_q = C_q , 
                                   C_a = C_a , C_f=C_f , N=N )        


ax[0].plot( sim_time , simps( yout , x , axis=1) , linewidth=2 , linestyle='solid')
ax[1].plot( sim_time , simps( x*yout , x , axis=1) , linewidth=2, linestyle='solid' )
ax1.plot( x , y0 , linewidth=2 , linestyle='solid')


#==============================================================================
# Final touches
#==============================================================================
ax[1].axhline(y=simps( x*sol.x , x) , linewidth=2 , color = 'k')
ax[1].locator_params(axis='y',nbins=4)
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$M_1(t)$')

ax[0].axhline(y=simps( sol.x , x) , linewidth=2 , color = 'k')
ax[0].locator_params(axis='y',nbins=4)
ax[0].set_ylabel('$M_0(t)$')

ax1.set_ylabel('$u_*(x)$')
ax1.set_xlabel('$x$')


fig.savefig( '../images/instability_moments.png' , bbox_inches='tight' , dpi=400, facecolor='white')              
fig1.savefig( '../images/instability_initial.png' , bbox_inches='tight' , dpi=400, facecolor='white')          
  


