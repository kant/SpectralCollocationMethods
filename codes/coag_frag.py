# -*- coding: utf-8 -*-
#Created on Oct 4, 2016
#@author: Inom Mirzaev


from __future__ import division

import numpy as np
import time 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from functools import partial
from scipy.integrate import quad, odeint, simps 
from scipy.misc import derivative
from scipy.optimize import root
from scipy.special import beta 

x0 = 0
x1 = 1


def lagrange(N, x0=x0, x1=x1 , C_a=1.3, C_f = 1.0):

    #x = np.linspace(x0, x1, N+1)
    x = ( -np.cos( np.pi / N * np.arange(N+1) ) + 1 )*(x1-x0) / 2 + x0
    
    xx, yy = np.meshgrid(x, x)
    
    AA = yy-xx
    AA[AA==0]=1
    
    a = np.prod(AA, axis=1) 
   
    #mth lagrangian polynomial
    def phi(y, m):
        """Evaluates m-th Lagrangian polynomial at point y"""
        
        return np.prod( (y-x)[np.arange(len(x))!=m] )/a[m] 
    
    #Initialization of nonlinear part
    Uxy = np.zeros( [N+1, N+1, N+1] )
    
    #Integration matrix for integrals of between x_k and x_N
    #W1 = np.zeros( [N+1, N+1] )
    
    
    #Integration matrix for integrals of between x_0 and x_k
    #W2 = np.zeros( [N+1, N+1] )
    W = np.zeros(N+1)
    
    for kk in xrange(N+1):
        W[kk] = quad(phi, x[0] , x[-1] , args=(kk,) )[0]
        for ii in xrange(N+1):
            #if kk<=ii:
                #W1[kk, ii] = quad(phi, x[kk] , x[-1] , args=(ii,) )[0]               
            if kk>=ii:
                #W2[kk, ii] = quad(phi, x[0] , x[kk] , args=(ii,) )[0]
                
                dummy = x[kk] - x[ii] - xx
                dummy[np.diag_indices(N+1)]=1
                Uxy[kk, ii,:] = np.prod( dummy, axis=1)/a
    

    #Aggregation in    
    a_ker = np.vectorize( partial( agg_kernel , C_a = C_a ) )
    
    Ain = 0.5*a_ker( yy - xx , xx )
    
    Ain[0]=0    
    Ain=Ain*W

    #Aggregation out    
    Aout = a_ker(yy, xx-yy)
    Aout[-1]=0
    
    Aout = Aout*W
   
    
    #Initialization of fragmentation in
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )
    
    Fin = gam( yy , xx )    
    Fin[-1] = 0 
    
    Fin = np.multiply( Fin*W, frag( x ) )         
    Fin[np.diag_indices(N+1)]*=0.5
    
  
    return x, Fin, Ain, Aout, Uxy

    
    
#Aggregation rate
def agg_kernel( x , y , C_a=1.3  ):
    if x<=0 or y<=0:
        return 0
    else:
        return C_a*( x ** ( 1/3 ) + y ** ( 1/3 ) ) **3      
        #return C_a*(x+y)      

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
    
def nonlinear_RHS(y, A , Ain, Aout, Uxy):
    
    Agg = np.dot(Uxy, y)
    dy = np.dot( Ain*Agg +A , y ) - np.sum( Aout*Agg.T , axis=1)*y
    
    return dy
    
    
def nonlinear_root(N,  C_a = 1.3 , C_f=1.0):
    
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )

    x, Fin, Ain, Aout, Uxy = lagrange(N , C_a=C_a, C_f = C_f)
           
    A = Fin -np.diag( 0.5*frag(x) )

    #myfunc = partial( nonlinear_RHS, A=A[1:, 1:] , Ain=Ain[1:, 1:], Aout=Aout[1:, 1:], Uxy=Uxy[1:, 1:,1:] ,g0=g( x[0] ) ) 
    
    myfunc = partial( nonlinear_RHS, A=A , Ain=Ain, Aout=Aout, Uxy=Uxy ) 
    
    seed = 1* np.ones(N+1)                    
    #sol = fsolve( myfunc ,  seed , xtol = 1e-12 , full_output=1 )
    
    sol = root( myfunc ,  seed , method='krylov' , options={'fatol':1e-8, 'maxiter':1000 ,'disp': False})
    
    
    print N, np.linalg.norm( myfunc(sol.x) )
    
    return sol, x
    
   
def simulate_ode( sim_time , y0 , C_a = 1.3 , C_f=1.0 , N=50 ):
    
        
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )
    
    
    x, Fin, Ain, Aout, Uxy = lagrange(N , C_a = C_a , C_f = C_f)
           
    A = Fin -np.diag( 0.5*frag(x) )

    def ode_renewal(y, t):
        
        Agg = np.dot(Uxy, y)
        dy = np.dot( Ain*Agg + A , y ) - np.sum( Aout*Agg.T , axis=1)*y
        
        return dy
 

    yout = odeint( ode_renewal , y0 , sim_time )
      
    return yout, sim_time , x

if __name__=='__main__':    
    
    start =  time.time()
    
 
    N = 100     
   
    G = 10
    C_a = 10#1.3 * G    
    C_f = 20#G  
   
    """
    sim_time = np.linspace(0, 100, 500)
    
                
    plt.close('all')
    
    fig, ax = plt.subplots( nrows=2 , ncols=1 , sharex=True  )    
    
    #==============================================================================
    # Initial condition #1
    #==============================================================================
  
    y0 = 1*np.ones(N+1)
    
    #y0 = np.min( sol.x )*np.random.uniform( -1, 1 , len(x) ) + sol.x
    
    yout, sim_time , x = simulate_ode( sim_time , y0 , C_a = C_a , C_f=C_f , N=N )        
    
    
    ax[0].plot( sim_time , simps( yout , x , axis=1) , linewidth=2  )
    ax[1].plot( sim_time , simps( x*yout , x , axis=1) , linewidth=2 )

    
    #==============================================================================
    # Final touches
    #==============================================================================
   
    ax[1].locator_params(axis='y',nbins=4)
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$M_1(t)$')
    
    ax[0].locator_params(axis='y',nbins=4)
    ax[0].set_ylabel('$M_0(t)$')"""
    
    

    #==============================================================================
    #  Coagulation fragmentation steady states   
    #==============================================================================
    
    sol , x = nonlinear_root( 40 , C_a=C_a, C_f=C_f )

    plt.figure()
    plt.plot( x , sol.x , linewidth=2 )
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel( '$p_*(x)$' )
    

    end = time.time()
    
    
    print "Elapsed time", round( end - start   , 2 ) ,  "seconds "


