# -*- coding: utf-8 -*-
#Created on Oct 4, 2016
#@author: Inom Mirzaev


from __future__ import division

import numpy as np
import time 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from functools import partial
from scipy.integrate import quad, odeint, simps, trapz , cumtrapz
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

    np.seterr(all='ignore')
    D = 1/AA
    np.seterr(all='raise')
    
    D = D -np.diag( np.diag(D))
    
    D_diag = np.diag( np.sum( D , axis=1) )
    
    D = np.multiply( np.multiply( D.T, a ).T , 1/a) + D_diag
    
    #Initialization of nonlinear part
    Uxy = np.zeros( [N+1, N+1, N+1] )
    
    #Integration matrix for integrals of between x_k and x_N
    #W1 = np.zeros( [N+1, N+1] )
    
    
    #Integration matrix for integrals of between x_0 and x_k
   
    for kk in xrange(N+1):

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
   
    #Aggregation out
    
    Aout = a_ker(yy, xx-yy)
    Aout[-1]=0
    
    #Initialization of fragmentation in
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )
    
    Fin = gam( yy , xx )    
    Fin[-1] = 0 
    
    Fin = np.multiply( Fin, frag( x ) )         
    #Fin[np.diag_indices(N+1)]*=0.5
      
    return x, D, Fin, Ain, Aout, Uxy

    
    
#Aggregation rate
def agg_kernel( x , y , C_a=1.3  ):
    if x<0 or y<0:
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
    
def nonlinear_RHS(y, A , Fin, Ain, Aout, Uxy, g0, x):
    
    #y[0] = simps(y*renewal(x, C_q=5), x)/g0
    y[0] = 1/g0
    Agg = np.dot(Uxy, y)

    dy = simps( Ain*Agg*y + Fin*y - ( (Aout*Agg.T).T*y ).T , x , axis=1 ) + np.dot( A  , y )

    #dy[1:] = 0.5*( dummy[1:] + dummy[:-1])
    #dy[1:] =  dy[:-1]
    dy[0] = 0
    
    return dy
    
    
def nonlinear_root(N, C_g, C_mu ,  C_a = 1.3 , C_f=1.0):
    
    g = np.vectorize( partial( growth , C_g=C_g ) )
    
    #Initialize removal function with paramter C_mu        
    rem = np.vectorize( partial( removal , C_mu=C_mu ) )
    
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )

    x, D, Fin, Ain, Aout, Uxy = lagrange(N , C_a=C_a, C_f = C_f)
           
    #A = -( D.T*g(x) ).T -np.diag(derivative( g , x, n=1, dx=1e-8) ) - np.diag( 0.5*frag(x)+rem(x) )

    A = -D*g(x) - np.diag( 0.5*frag(x)+rem(x) )

    #myfunc = partial( nonlinear_RHS, A=A[1:, 1:] , Ain=Ain[1:, 1:], Aout=Aout[1:, 1:], Uxy=Uxy[1:, 1:,1:] ,g0=g( x[0] ) ) 
    
    myfunc = partial( nonlinear_RHS, A=A , Fin=Fin, Ain=Ain, Aout=Aout, Uxy=Uxy, g0=g( x[0] ) , x=x ) 
    
    seed = 1/g(x[0])* np.ones(N+1)                    
    #Krylov options
    #opts = {'fatol':1e-8, 'maxiter':1000 ,'disp': False}
    
    sol = root( myfunc ,  seed , method='hybr' , options={'xtol':1e-6})
    
    
    print N, np.linalg.norm( myfunc(sol.x) )
    
    return sol, x
    

def renewal_RHS(y, A , Fin, Ain, Aout, Uxy, g0, x, ren):
    
    y[0] = simps(y*ren, x)/g0
    Agg = np.dot(Uxy, y)
    dy = simps( Ain*Agg*y + Fin*y - ( (Aout*Agg.T).T*y ).T , x , axis=1 ) + np.dot( A  , y )
    #dy[1:] =  dy[:-1]
    dy[0] = 0
 
    return dy
    
    
def renewal_root(N, C_g=1, C_mu=np.exp(-1) , C_q=4,  C_a = 1.3 , C_f=1.0):
    
    g = np.vectorize( partial( growth , C_g=C_g ) )
    
    #Initialize removal function with paramter C_mu        
    rem = np.vectorize( partial( removal , C_mu=C_mu ) )
    
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )

    q = np.vectorize( partial( renewal , C_q = C_q ) )

    x, D, Fin, Ain, Aout, Uxy = lagrange(N , C_a=C_a, C_f = C_f)
           
    #A = -( D.T*g(x) ).T -np.diag(derivative( g , x, n=1, dx=1e-8) ) - np.diag( 0.5*frag(x)+rem(x) )

    A = -D*g(x) - np.diag( 0.5*frag(x)+rem(x) )

    #myfunc = partial( nonlinear_RHS, A=A[1:, 1:] , Ain=Ain[1:, 1:], Aout=Aout[1:, 1:], Uxy=Uxy[1:, 1:,1:] ,g0=g( x[0] ) ) 
    
    myfunc = partial( renewal_RHS, A=A , Fin=Fin, Ain=Ain,
                     Aout=Aout, Uxy=Uxy, g0=g( x[0] ) , x=x , ren=q(x) ) 
    
    seed = 1* np.ones(N+1)                    
    
    sol = root( myfunc ,  seed , method='hybr' , options={'xtol':1e-6})
    
    
    print N, np.linalg.norm( myfunc(sol.x) )
    
    return sol, x
       
def simulate_ode( sim_time , y0 ,  C_g=1 , C_mu=0.01 , C_q = 1 , C_a = 1.3 , C_f=1.0 , N=50 ):
    
    g = np.vectorize( partial( growth , C_g=C_g ) )
    
    #Initialize removal function with paramter C_mu        
    rem = np.vectorize( partial( removal , C_mu=C_mu ) )
    
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )
    
    q = np.vectorize( partial( renewal , C_q = C_q ) )
    
    x, D, Fin, Ain, Aout, Uxy = lagrange(N , C_a = C_a , C_f = C_f)
           
    #A = -( D.T*g(x) ).T -np.diag(derivative( g , x, n=1, dx=1e-2) ) -np.diag( 0.5*frag(x)+rem(x) )
    A = -D*g(x) - np.diag( 0.5*frag(x)+rem(x) )

    def ode_renewal(y, t):
        
        y[0] = simps( y*q(x) , x ) / g( x[0] )
        
        #y[0] = 1 / g( x[0] )
        
        Agg = np.dot(Uxy, y)
        #dy = np.zeros_like(y)
      
        dy = simps( Ain*Agg*y + Fin*y - ( (Aout*Agg.T).T*y ).T , x , axis=1 ) + np.dot( A  , y )
      
        #dy  = dummy
        
        #dy[0] = 0
        #dy[1:] = 0.5*( dummy[1:] + dummy[:-1])
        #dy[0] = dummy[0] 
        
        #dy[0] = ( y[1] - y[0] ) / (x[1] - x[0] )
        
        return dy
 

    yout = odeint( ode_renewal , y0 , sim_time )
      
    return yout, sim_time , x

 
    
if __name__=='__main__':    
    
    start = time.time()
    
    C_g = 1

    G = 10
    C_a = 1.3 * G    
    C_f = G  
    C_mu =np.exp(-G) 
    
    
    #==============================================================================
    # Nonlinear case
    #==============================================================================
    
    from scipy import interpolate
    
    
    sol , x = nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f )
    
    interp_func  = interpolate.interp1d( x , sol.x,  kind='quadratic')
    
    
    dims = np.arange( 10 , 110 , 10  )
    max_err = []
    
    for N in dims:
        sol , x = nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f )
        
        appr_sol = sol.x    
        actual_sol = interp_func( x )
        
        max_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )
    
    plt.close('all')
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
    plt.plot( x, sol.x )
    
    

    
    end = time.time()
    
    
    print "Elapsed time", round( end - start   , 2 ) ,  "seconds "


