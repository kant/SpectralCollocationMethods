# -*- coding: utf-8 -*-
#Created on Feb 18, 2016
#@author: Inom Mirzaev

"""
    Model rates and parameters used for generation of existence and stability maps
    of the population balance equations (PBEs) (see Mirzaev, I., & Bortz, D. M. (2015). 
    arXiv:1507.07127 ). 
"""


from __future__ import division
from functools import partial

import scipy.linalg as lin
import numpy as np
from scipy.special import beta
from scipy.optimize import root
import matplotlib.pyplot as plt
import time
"""
    Number of CPUs used for computation of existence and stability regions.
    For faster computation number of CPUs should be set to the number of cores available on
    your machine.
"""

ncpus = 3

# Minimum and maximum floc sizes
x0 = 0
x1 = 1

#Aggregation rate
def agg_kernel( x , y , C_a=1.3, x1=x1  ):
    if x<0 or y<0 or (x+y)>x1:
        return 0
    else:
        return C_a*( x ** ( 1/3 ) + y ** ( 1/3 ) ) **3      
        #return C_a*(x+y)      

def renewal(x, C_q=1):
    
    return C_q*( x**(2/3) )

#Growth rate   
def growth(x ,  C_g, x1=x1 ):
    
    return C_g*(x+1)#C_g*x*(x1-x)
            
#Removal rate    
def removal( x , C_mu ):
     #Should return a vector
     return C_mu*x


def gam( y , x , a=2, b=2):
    
    """Post-fragmentation density distribution"""
    
    if y>x or x==0:
        return 0
    else:       
        return y**(a-1) * ( np.abs( x - y )**(b-1) )  / ( x**(a+b-1) ) / beta( a , b ) 
     

gam = np.vectorize( gam )        
    
#Fragmentation rate
def fragmentation( x , C_f=1.0):
    
    return 0.0007*C_f**1.6*x**(1/3)



"""
    Parameters used for stability plots in 'pbe_stability_plots.py' and 
    eigenvalue plots in 'pbe_jacobian_eigenvalue_plots.py'
"""
a = 1
b = 0.5
c = 1


    
#Initializes uniform partition of (x0, x1) and approximate operator F_n
def initialization(N , C_g, C_mu ,  C_a = 1.3 , C_f=1.0):
    
    #delta x
    dx = ( x1 - x0 ) / N
    
    #Uniform partition into smaller frames
    nu = x0 + np.arange(N+1) * dx
    
    #Aggregation in
    Ain = np.zeros( ( N , N ) )
    
    #Aggregation out
    Aout = np.zeros( ( N , N ) )
    
    #Fragmentation in
    Fin = np.zeros( ( N , N ) )
    
    #Fragmentation out
    Fout = np.zeros( N )

    #Re-initialize growth function with parameter b
    grow = partial( growth , C_g=C_g )
    
    renew = partial( renewal , C_q=1/grow(nu[0]))
    
    #Re-initialize removal rate with paramter c        
    rem = partial( removal , C_mu=C_mu )
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )
    a_ker = np.vectorize( partial( agg_kernel , C_a = C_a ) )
    
    #Initialize matrices Ain, Aout and Fin
    for mm in range( N ):
    
        for nn in range( N ):
            
            if mm>nn:
            
                Ain[mm,nn] = 0.5 * dx * a_ker( nu[mm] , nu[nn+1] )
            
            if mm + nn < N-1 :
                
                Aout[mm, nn] = dx * a_ker( nu[mm+1] , nu[nn+1] )
                    
            if nn > mm :
            
                Fin[mm, nn] = dx * gam( nu[mm+1], nu[nn+1] ) * frag( nu[nn+1] )


    #Initialize matrix Fout
    Fout = 0.5 * frag( nu[range( 1 , N + 1 ) ] ) + rem( nu[range( 1 , N + 1 )] )

    #Growth matrix
    
    Gn = -np.diag( grow( nu[range( 1 , N + 1 )] ) ) + np.diag( grow( nu[range( 1 , N)] )  , k=-1)    
    Gn /= dx
    
    #Gn[0,:] = Gn[0,:] + renew( nu[range( 1 , N+1 ) ] )
    
    #Growth - Fragmentation out + Fragmentation in
    An = Gn - np.diag( Fout ) + Fin
    
    return (An , Ain , Aout , nu , N , dx)



#Approximate operator for the right hand side of the evolution equation
def approximate_IG( y ,  An , Aout , Ain ,g0 ):
    
    y[0] = 1/g0
    a = np.zeros_like(y)

    a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    

    out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T - ( Aout.T * y ).T + An , y ) 
    out[0] =0
    
    return out


#Exact Jacobian of the RHS 
def jacobian_IG(y, An , Aout , Ain):

    a = np.zeros_like(y)

    a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ] 

    out = An - ( Aout.T * y ).T - np.diag( np.dot(Aout , y) ) + 2*Ain * lin.toeplitz( np.zeros_like(y) , a).T

    return out    
 
    
def nonlinear_root(N, C_g, C_mu ,  C_a = 1.3 , C_f=1.0):

    
    g = np.vectorize( partial( growth , C_g=C_g ) )
    
    An, Ain, Aout, nu, N, dx = initialization( N ,  C_g, C_mu ,  C_a = C_a, C_f=C_f )
    
    root_finding  = partial( approximate_IG , An=An, Aout=Aout, Ain=Ain, g0=g(nu[0]) )    

    
    seed = 1/g( nu[0] )* np.ones(N)                    
    #Krylov options
    #opts = {'fatol':1e-8, 'maxiter':1000 ,'disp': False}
    
    #Hybrid options
    opts = {'xtol':1e-6}
    
    sol = root( root_finding ,  seed , method='hybr' , options=opts)
    
    
    print N, np.linalg.norm( root_finding(sol.x) )
    
    #sol.x = sol.x/sol.x[0]
    
    return sol, nu[:-1]
    
if __name__=='__main__':    
    
    start = time.time()
    
    C_g = 1

    G = 1
    C_a = 1.3 * G    
    C_f = G  
    C_mu =np.exp(-G) 

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
    
    logx = np.log(  1/dims )
    logy = np.log( max_err )
    
    coeffs = np.polyfit( logx , logy ,deg=1 )
    plt.title('Convergence order= '+str( round(coeffs[0], 2) ) )
    
   

    plt.figure()
    plt.plot(x, sol.x )    
    end = time.time()
    
    
    print "Elapsed time", round( end - start   , 2 ) ,  "seconds "

