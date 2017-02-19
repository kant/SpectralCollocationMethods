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


C_g = 1

G = 1
C_a = 1.3 * G    
C_f = G  
C_mu =np.exp(-G) 

#==============================================================================
# Linear case
#==============================================================================

dims = np.arange(5, 55, 5)
max_err = []

for N in dims:     
    x, exact_sol , approx_sol = linear_root(N , C_g , C_mu) 
    max_err.append( np.max( np.abs( np.ravel( approx_sol ) - exact_sol ) ) )


max_err = np.asarray( max_err )


plt.close('all')

plt.figure()
plt.grid(True)
plt.semilogy(dims, max_err , linewidth=1, marker='o', markersize=10 )
plt.xlabel('Approximation dimension ($N$)' )
plt.ylabel( '$\Vert u_* - u_*^N  \Vert_{\infty}$' )

yticks = plt.yticks()[0]
plt.yticks( yticks[::2] )

plt.savefig( '../images/error_linear_case.png' , bbox_inches='tight' , dpi=400,facecolor='white')




#==============================================================================
# Nonlinear case
#==============================================================================

from scipy import interpolate


sol , x = nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f )

interp_func  = interpolate.interp1d( x , sol.x,  kind='quadratic')


dims = np.arange( 10 , 110 , 10 )
max_err = []

for N in dims:
    sol , x = nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f )
    
    appr_sol = sol.x    
    actual_sol = interp_func( x )
    
    max_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )


plt.figure()
plt.grid(True)
plt.semilogy(dims, max_err , linewidth=1, marker='o', markersize=10 )
#myaxes = list( plt.axis() )
#myaxes[0]-=5
#myaxes[1]+=5
#plt.axis( myaxes )


plt.xlabel('Approximation dimension ($N$)' )
plt.ylabel( '$\Vert u_*^{200} - u_*^N  \Vert_{\infty}$' )
plt.savefig( '../images/error_nonlinear_case.png' , bbox_inches='tight' , dpi=400, facecolor='white')



#==============================================================================
# Comparison of integral approximations
#==============================================================================

import simpsons_rule, gauss_rule, trapezoidal_rule, old_method

sol , x = simpsons_rule.nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f )
interp_simps  = interpolate.interp1d( x , sol.x,  kind='quadratic' )

sol , x = gauss_rule.nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f )
interp_gauss  = interpolate.interp1d( x , sol.x,  kind='quadratic' )

sol , x = trapezoidal_rule.nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f )
interp_trapz  = interpolate.interp1d( x , sol.x,  kind='quadratic' )

sol , x = old_method.nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f )
interp_old  = interpolate.interp1d( x , sol.x,  kind='quadratic' )


dims = np.arange( 5 , 45 , 5 )

simps_err = []
trapz_err = []
gauss_err = []
old_err = []

for N in dims:
    sol , x = simpsons_rule.nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f ) 
    appr_sol = sol.x    
    actual_sol = interp_simps( x ) 
    simps_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )

    sol , x = gauss_rule.nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f ) 
    appr_sol = sol.x    
    actual_sol = interp_gauss( x ) 
    gauss_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )


    
    sol , x = trapezoidal_rule.nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f ) 
    appr_sol = sol.x    
    actual_sol = interp_trapz( x ) 
    trapz_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )


    sol , x = old_method.nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f ) 
    appr_sol = sol.x    
    actual_sol = interp_old( x ) 
    old_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )
    
    
plt.figure()
plt.grid(True)

logx = np.log(  1/dims )
logy = np.log( trapz_err )
coeffs = np.polyfit( logx , logy ,deg=1 )
plt.loglog(1/dims, trapz_err , linewidth=1, marker='v' , 
           markersize=10 , label='Trapezoidal\nconv. rate= '+str(round(coeffs[0],2) ) )


logx = np.log(  1/dims )
logy = np.log( simps_err )
coeffs = np.polyfit( logx , logy ,deg=1 )
plt.loglog(1/dims, simps_err , linewidth=1, marker='*' , 
           markersize=10 , label='Simpson\'s\nconv. rate= '+str(round(coeffs[0],2) ) )


logx = np.log(  1/dims )
logy = np.log( gauss_err )
coeffs = np.polyfit( logx , logy ,deg=1 )
plt.loglog(1/dims, gauss_err , linewidth=1, marker='o' , 
           markersize=10 , label='Gauss\nconv. rate= '+str(round(coeffs[0], 2 ) ) )


#myaxes = list( plt.axis() )
#myaxes[0]-=5
#myaxes[1]+=5
#plt.axis( myaxes )

plt.xlabel('Grid size ($\Delta x$)' )
plt.ylabel( '$\Vert u_*^{200} - u_*^N  \Vert_{\infty}$' )

plt.legend(bbox_to_anchor=(1.4, 0.75), bbox_transform=plt.gcf().transFigure , 
           fancybox=True, frameon=True)

plt.savefig( '../images/conv_rate_integrals.png' , bbox_inches='tight' , dpi=400, facecolor='white')





#==============================================================================
# Comparison of integral approximations
#==============================================================================


dims = np.arange( 10 , 160 , 10 )

simps_err = []
trapz_err = []
gauss_err = []
old_err = []

for N in dims:
    sol , x = simpsons_rule.nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f ) 
    appr_sol = sol.x    
    actual_sol = interp_simps( x ) 
    simps_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )

    sol , x = gauss_rule.nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f ) 
    appr_sol = sol.x    
    actual_sol = interp_gauss( x ) 
    gauss_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )


    
    sol , x = trapezoidal_rule.nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f ) 
    appr_sol = sol.x    
    actual_sol = interp_trapz( x ) 
    trapz_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )


    sol , x = old_method.nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f ) 
    appr_sol = sol.x    
    actual_sol = interp_old( x ) 
    old_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )
    
    
plt.figure()
plt.grid(True)

plt.semilogy(dims, trapz_err , linewidth=1, marker='v' , 
            markersize=10 , label='Trapezoidal' )


plt.semilogy(dims, simps_err , linewidth=1, marker='*' , 
            markersize=10 , label='Simpson\'s' )


plt.semilogy(dims, gauss_err , linewidth=1, marker='o' , 
            markersize=10 , label='Gauss' )

plt.xlabel('Approximation dimension ($N$)' )
plt.ylabel( '$\Vert u_*^{200} - u_*^N  \Vert_{\infty}$' )

plt.legend(bbox_to_anchor=(1.3, 0.7), bbox_transform=plt.gcf().transFigure , 
           fancybox=True, frameon=True)

plt.savefig( '../images/error_integrals.png' , bbox_inches='tight' , dpi=400, facecolor='white')


#==============================================================================
# Comparison of Chapter 3 and Chapter 4 methods
#==============================================================================
plt.figure()
plt.grid(True)

logx = np.log(  1/dims )
logy = np.log( old_err )
coeffs = np.polyfit( logx , logy ,deg=1 )
plt.semilogy(dims, old_err , linewidth=1, marker='*' , markersize=10 , 
           label='Chapter 3 method' )


logx = np.log(  1/dims )
logy = np.log( gauss_err )
coeffs = np.polyfit( logx , logy ,deg=1 )
plt.semilogy(dims, gauss_err , linewidth=1, marker='o' , markersize=10 , 
           label='Collocation with Simpson\'s' )

plt.legend()


#myaxes = list( plt.axis() )
#myaxes[0]-=5
#myaxes[1]+=5
#plt.axis( myaxes )

plt.xlabel('Approximation dimension ($N$)' )
plt.ylabel( '$\Vert u_*^{200} - u_*^N  \Vert_{\infty}$' )
plt.savefig( '../images/prev_with_imporved.png' , bbox_inches='tight' , dpi=400, facecolor='white')



#==============================================================================
# Example steady state solution: Linear and nonlinear together
#==============================================================================

x, exact_sol , approx_sol = linear_root( 30 , C_g, C_mu)

sol , x = nonlinear_root( 30 , C_g , C_mu, C_a=C_a, C_f=C_f )


plt.figure()
plt.plot( x , approx_sol , linewidth=3 , label='Linear' ,
                color = sns.xkcd_rgb["pale red"] , 
                linestyle='--')
plt.plot( x , sol.x , linewidth=2 , 
         label='Nonlinear', 
         color = sns.xkcd_rgb["denim blue"])
plt.legend()
plt.xlabel('$x$')
plt.ylabel( '$u_*(x)$' )

plt.savefig( '../images/example_solution.png' , bbox_inches='tight' , dpi=400, facecolor='white')

#==============================================================================
# Increasing growth rate
#==============================================================================

Cg_values = [0.1, 1 , 20]
fig , ax = plt.subplots( nrows=3 , ncols=1 , sharex=True )

for pp in range(3):
    
    x, exact_sol , approx_sol = linear_root( 50 , Cg_values[pp] , C_mu)
    
    sol , x = nonlinear_root( 50 , Cg_values[pp] , C_mu , C_a=C_a, C_f=C_f )

    ax[pp].plot( x , sol.x , linewidth=3 , 
                label='Nonlinear' ,  
                color = sns.xkcd_rgb["denim blue"] )
    ax[pp].plot( x , approx_sol , linewidth=2 , label='Linear' , 
                color = sns.xkcd_rgb["pale red"] , 
                linestyle='--')
    
    ax[pp].set_ylabel( '$u_*(x)$' )
    ax[pp].locator_params(axis='y',nbins=3)
    #yticks = ax[pp].get_yticks()
    #new_ticks = [0 , int( 0.5*np.max(yticks) ) ,  np.max(yticks) ]
    #ax[pp].set_yticks( new_ticks )
    
    ax[pp].text(0.8, 0.7, '$C_g='+str(Cg_values[pp] ) +'$' , 
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax[pp].transAxes)
    
    if pp==2:
        ax[pp].set_xlabel('$x$')
    if pp==0:
       lgd = ax[pp].legend( loc='upper center', bbox_to_anchor=(0.5, 1.5),
              fancybox=True, shadow=True, ncol=5 )

        
plt.savefig( '../images/growth_increasing.png' , bbox_extra_artists=(lgd,) , 
            bbox_inches='tight' , dpi=400, facecolor='white')




#==============================================================================
# Decreasing shear rate
#==============================================================================

G_values = [10, 1 , 0.01]
fig , ax = plt.subplots( nrows=3 , ncols=1 , sharex=True )

for pp in range(3):
    
    G = G_values[pp]
    C_a = 1.3 * G    
    C_f = G  
    C_mu = np.exp(-G)
    
    x, exact_sol , approx_sol = linear_root( 50 , C_g , C_mu)
    
    sol , x = nonlinear_root( 50 , C_g , C_mu , C_a=C_a, C_f=C_f )

    ax[pp].plot( x , sol.x , linewidth=3 , 
                label='Nonlinear' ,  
                color = sns.xkcd_rgb["denim blue"] )
    ax[pp].plot( x , approx_sol , linewidth=2 , label='Linear' , 
                color = sns.xkcd_rgb["pale red"] , 
                linestyle='--')
    
    ax[pp].set_ylabel( '$u_*(x)$' )
    ax[pp].locator_params(axis='y',nbins=3)
    #yticks = ax[pp].get_yticks()
    #new_ticks = [0 , int( 0.5*np.max(yticks) ) ,  np.max(yticks) ]
    #ax[pp].set_yticks( new_ticks )
    
    ax[pp].text(0.8, 0.7, '$\dot{\gamma}='+str(G_values[pp] ) +'$' , 
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax[pp].transAxes)
    
    if pp==2:
        ax[pp].set_xlabel('$x$')
    if pp==0:
       lgd = ax[pp].legend( loc='upper center', bbox_to_anchor=(0.5, 1.5),
              fancybox=True, shadow=True, ncol=5 )

        
plt.savefig( '../images/shear_decreasing.png' , bbox_extra_artists=(lgd,) , 
            bbox_inches='tight' , dpi=400, facecolor='white')


