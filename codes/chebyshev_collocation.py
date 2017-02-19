# -*- coding: utf-8 -*-
#Created on Oct 4, 2016
#@author: Inom Mirzaev


from __future__ import division

import numpy as np
import time , sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from functools import partial
from scipy.integrate import quad, odeint, simps, trapz  
from scipy.special import beta 
from scipy.optimize import root
import spectral_collocation as sc
from scipy.misc import derivative


x0 = 0 

x1 = 1


def cheb(N, x0=x0, x1=x1 , C_a=1.3, C_f = 1.0):

    x = np.cos( np.pi / N * np.arange(N+1) ).reshape( [N+1, 1] )
    
    c = np.ones( N+1 )  
    c[0]=2
    c[-1] =2  
    c = c* ( (-1)**np.arange(N+1) )  
    c = np.reshape( c,  [N+1, 1] )
    
    X = np.kron(np.ones( [1 , N+1] )  , x )  
    dX = X - X.T
    
    D = np.dot( c , ( 1 /c).T ) /( dX + np.eye(N+1) ) 
    D = 2/(x1-x0)*( D - np.diag( np.sum( D , axis=1 ) ) )
    D = np.flipud( np.fliplr( D ) )
    
    #x = 0.5*(x1-x0)*np.ravel(x) + 0.5*(x1+x0)
    #x = ( -np.cos( np.pi / N * np.arange(N+1) ) + 1 )*(x1-x0) / 2 + x0
    x = 0.5*(x1-x0)*np.ravel(x)[::-1] + 0.5*(x1+x0)
    xx, yy = np.meshgrid(x, x)
    
    
    #mth Chebyshev polynomial
    def phi(y, m):
        
        if y<0:
            return 0
        else:
            return np.cos( m*np.arccos( (y - 0.5*(x1+x0) )*2/(x1-x0) ) )
                       
    phi = np.vectorize( phi )      
    
    #Initialization of nonlinear part
    Uxy = np.zeros( [N+1, N+1, N+1] )
    """
    for kk in xrange(N+1):
        for ii in xrange(N+1):
            if kk>=ii:
                Uxy[kk, ii, :] = phi(x[kk] - x[ii] , np.arange(N+1) )
    """
    Fxy = np.zeros( [N+1, N+1] )            
    for jj in xrange(N+1):
        Fxy[jj, :] =  phi(x[jj] , np.arange(N+1) )
        Uxy[:, :, jj] = phi(yy-xx, jj)
    
        
    #Aggregation in    
      
    a_ker = np.vectorize( partial( agg_kernel , C_a = C_a ) )
    
    Ain = 0.5*a_ker( yy - xx , xx )
    
    Ain[0]=0
   
    #Aggregation out
    
    Aout = a_ker( yy , xx-yy )
    Aout[-1]=0
    
    #Initialization of fragmentation in
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )
    
    Fin = gam( yy , xx )    
    Fin[-1] = 0 
    
    Fin = np.multiply( Fin , frag( x ) )       
    
    return x, D, Fin, Ain, Aout, Uxy
    

 
#Aggregation rate
def agg_kernel( x , y , C_a=1.3  ):
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


def IC(x, x1=x1, x0=x0):
    
    return 1*x*(x1-x) 
    
def linear_root(N, C_g , C_mu):

    g = np.vectorize( partial( growth , C_g=C_g ) )
    
    #Initialize removal function with paramter C_mu        
    rem = partial( removal , C_mu=C_mu )  
    
    # Removal over growth for integration  
    def mug_int(x):
           
        return - rem(x) / g(x)
        
    x, D, Fin, Ain, Aout, Uxy = cheb(N)
           
    A = - D*g(x) - np.diag( rem(x) )
         
    approx_sol = np.ones(N+1)
     
    approx_sol[1:] = np.linalg.solve( A[1:, 1:]  , -A[1:, 0] ) 
    
    approx_sol *= 1 / g( x[0] )
        
    #Exact steady state solution
    yy= np.zeros_like( x )
    
    for num in range( len(yy) ):    
        yy[num] = quad( mug_int , 0 , x[num] )[0]
        
    exact_sol =  1 / g( x ) * np.exp( yy )

    return x, exact_sol , approx_sol    

    
    
def nonlinear_RHS(y, A , Fin, Ain, Aout, Uxy, g0, x):
    
    y[0] = 1/g0
    Agg = np.dot(Uxy, y)
    dy = np.zeros_like(y)
  
    dummy = trapz( Ain*Agg*y + Fin*y - ( (Aout*Agg.T).T*y ).T , x , axis=1 ) + np.dot( A  , y )
  
    #dy[1:] = 0.5*( dummy[1:] + dummy[:-1])
    dy[1:] =  dummy[:-1]
        
    return dy
    
    
def nonlinear_root(N, C_g, C_mu ,  C_a = 1.3 , C_f=1.0):
    
    g = np.vectorize( partial( growth , C_g=C_g ) )
    
    #Initialize removal function with paramter C_mu        
    rem = np.vectorize( partial( removal , C_mu=C_mu ) )
    
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )

    x, D, Fin, Ain, Aout, Uxy = cheb(N , C_a=C_a, C_f = C_f)
     
    A = - D*g(x) - np.diag( 0.5*frag(x)+ rem(x) )      
    #A = -( D.T*g(x) ).T -np.diag(derivative( g , x, n=1, dx=1e-8) ) - np.diag( 0.5*frag(x)+rem(x) )

    #myfunc = partial( nonlinear_RHS, A=A[1:, 1:] , Ain=Ain[1:, 1:], Aout=Aout[1:, 1:], Uxy=Uxy[1:, 1:,1:] ,g0=g( x[0] ) ) 
    
    myfunc = partial( nonlinear_RHS, A=A , Fin=Fin, Ain=Ain, Aout=Aout, Uxy=Uxy, g0=g( x[0] ) , x=x ) 
    
    seed = 1/g(x[0])* np.ones(N+1)                    
    #Krylov options
    #opts = {'fatol':1e-8, 'maxiter':1000 ,'disp': False}
    
    #Hybrid options
    opts = {'xtol':1e-6}
    
    sol = root( myfunc ,  seed , method='hybr' , options=opts)
    
    
    print N, np.linalg.norm( myfunc(sol.x) )
    
    return sol, x
    
if __name__=='__main__':
    
    start =  time.time()
    N= 30
    
    C_g = 1
    
    G = 10
    C_a = 1.3 * G    
    C_f = G  
    C_mu =np.exp(-G) 
    
    
    x, exact_sol , approx_sol = linear_root(N, C_g, C_mu)
    
    
    plt.close('all')
    
    
    plt.figure()
    
    plt.plot(x, np.abs( exact_sol - approx_sol ) )
    
    
    plt.figure()
    
    plt.plot(x, approx_sol )
    
    
    
    #==============================================================================
    # Nonlinear case
    #==============================================================================
    
    from scipy import interpolate
    
    
    sol , x = sc.nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f )
    
    interp_func  = interpolate.interp1d( x , sol.x,  kind='linear')
    
    
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
    plt.ylabel( '$\Vert p_*^{100} - p_*^N  \Vert_{\infty}$' )

    plt.figure()    
    plt.plot( x, sol.x )
    
    end = time.time()
    
    
    print "Elapsed time", round( end - start   , 2 ) ,  "seconds "
    
    
