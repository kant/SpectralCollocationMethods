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
import scipy.linalg as lin

x0 = 0
x1 = 1


def lagrange(N, x0=x0, x1=x1 , C_a=1.3, C_f = 1.0):

    x = np.linspace(x0, x1, N+1)
    
    #Chebyshev-Gauss-Lobatto points
    #x = ( -np.cos( np.pi / N * np.arange(N+1) ) + 1 )*(x1-x0) / 2 + x0
                                    
    
    #Chebyshev nodes
    #x = 0.5*(x0+x1) +0.5*(x1-x0)*np.cos( ( 2*np.arange(N+1,0,-1)-1 )*np.pi/(N+1)/2 )
      
    xx, yy = np.meshgrid(x, x)
    
    #Aggregation in    
    a_ker = np.vectorize( partial( agg_kernel , C_a = C_a ) )
    
    Ain = 0.5*a_ker( yy - xx , xx )
    Ain[0]=0

    #Aggregation out
    
    Aout = a_ker(yy, xx)
    Aout = np.fliplr( Aout )
    Aout[np.diag_indices(N+1)]*=0.5
    Aout = np.fliplr( Aout )
    
    Aout[-1]=0   
    #Aout[:, 0] *=0.5
  
    
    #Initialization of fragmentation in
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )
    
    Fin = gam( yy , xx )    
    Fin[-1] = 0 
    
    Fin = np.multiply( Fin, frag( x ) )         
    Fin[np.diag_indices(N+1)]*=0.5
    #Fin[:,-1] *=0.5

          
    return x, Fin, Ain, Aout

    
    
#Aggregation rate
def agg_kernel( x , y , C_a=1.3, x1=x1  ):
    if x<0 or y<0 or (x+y)>x1:
        return 0.0
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
    
def nonlinear_RHS(y, A , Fin, Ain, Aout, g0, x, method='trapz'):
    
    dy  = np.zeros_like(y)
    
    y[0] = 1 / g0[0]    
    
    dummy = np.trapz( Ain * lin.toeplitz( np.zeros_like(y) , y).T*y 
                  -(Aout.T * y ).T*y + Fin*y , x, axis=1 )+ A*y

    #dummy = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , y).T - ( Aout.T * y ).T + Fin, y ) + A*y
    
    if method=='trapz':
        dy[1:] = 0.5*( dummy[:-1] + dummy[1:] ) + ( y[:-1]*g0[:-1] - y[1:]*g0[1:] )/(x[1:]-x[:-1]) 
    if method=='euler':
        dy[1:] = dummy[:-1]+(  y[:-1]*g0[:-1] - y[1:]*g0[1:] )/(x[1:]-x[:-1])
    if method=='back':
        dy[1:] = dummy[1:] +(  y[:-1]*g0[:-1] - y[1:]*g0[1:] )/(x[1:]-x[:-1])
    if method=='milne':
        dy[1] = 0.5*( dummy[0] + dummy[1] ) + ( y[0]*g0[0] - y[1]*g0[1] )/(x[1]-x[0]) 
        dy[2:] = ( dummy[:-2] + 4*dummy[1:-1] + dummy[2:])/3 +\
                 ( y[:-2]*g0[:-2]- y[2:]*g0[2:] )/(x[1]-x[0])
    if method=='AB':
       #dy[1] = 0.5*( dummy[0]/g0[0] + dummy[1]/g0[1] ) + ( y[0] - y[1] )/(x[1]-x[0]) 
       #dy[2:] = (x[1]-x[0])*0.5*( 3*dummy[1:-1]/g0[1:-1] - dummy[:-2]/g0[:-2] ) + ( y[1:-1] - y[2:] ) 
       
       #dy[1:3] = 0.5*( dummy[0:2]/g0[0:2] + dummy[1:3]/g0[1:3] ) + ( y[0:2] - y[1:3] )/(x[1]-x[0]) 
       dy[1] = 0.5*( dummy[0] + dummy[1] ) + ( y[0]*g0[0] - y[1]*g0[1] )/(x[1]-x[0])
       dy[2] = 0.5*( 3*dummy[1] - dummy[0] ) + ( y[1]*g0[1] - y[2]*g0[2] )/(x[1]-x[0]) 
       dy[3:] = 1/12*( 23*dummy[2:-1] -\
                 16*dummy[1:-2] +5*dummy[:-3] ) + ( y[2:-1]*g0[2:-1] - y[3:]*g0[3:] )/(x[1]-x[0]) 
        
 
    return dy
    
    
def nonlinear_root(N, C_g, C_mu ,  C_a = 1.3 , C_f=1.0 , method='trapz'):
    
    g = np.vectorize( partial( growth , C_g=C_g ) )
    
    #Initialize removal function with paramter C_mu        
    rem = np.vectorize( partial( removal , C_mu=C_mu ) )
    
    frag = np.vectorize( partial( fragmentation , C_f = C_f ) )

    x,  Fin, Ain, Aout = lagrange(N , C_a=C_a, C_f = C_f)
           
    #A = -derivative( g , x, n=1, dx=1e-8) - 0.5*frag(x)-rem(x) 
    A =  - 0.5*frag(x)-rem(x) 

    #myfunc = partial( nonlinear_RHS, A=A[1:, 1:] , Ain=Ain[1:, 1:], Aout=Aout[1:, 1:], Uxy=Uxy[1:, 1:,1:] ,g0=g( x[0] ) ) 
    
    myfunc = partial( nonlinear_RHS, A=A , Fin=Fin, Ain=Ain, Aout=Aout,  g0=g( x ) , x=x , method=method) 
    
    seed = 1/g( x[0] )* np.ones(N+1)                    
    #Krylov options
    #opts = {'fatol':1e-8, 'maxiter':1000 ,'disp': False}
    
    #Hybrid options
    opts = {'xtol':1e-6}
    
    sol = root( myfunc ,  seed , method='hybr' , options=opts)
    
    
    print N, np.linalg.norm( myfunc(sol.x) )
    
    return sol, x
    

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
    
    import simpsons_rule
    
    sol , x = nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f , method='euler' )    
    interp_euler  = interpolate.interp1d( x , sol.x,  kind='linear')
    
    sol , x = nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f , method='back' )    
    interp_back  = interpolate.interp1d( x , sol.x,  kind='linear')

    sol , x = nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f , method='trapz' )    
    interp_trapz  = interpolate.interp1d( x , sol.x,  kind='linear')

    sol , x = nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f , method='milne' )    
    interp_milne  = interpolate.interp1d( x , sol.x,  kind='linear')

    sol , x = nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f , method='AB' )    
    interp_AB = interpolate.interp1d( x , sol.x,  kind='linear')
    
    dims = np.arange( 10 , 110 , 10 )
    euler_err = []
    back_err = []
    trapz_err = []
    milne_err = []
    AB_err = []
    
    for N in dims:
        sol , x = nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f , method='euler' )       
        appr_sol = sol.x    
        actual_sol = interp_euler( x )       
        euler_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )
 
        
        sol , x = nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f , method='back' )       
        appr_sol = sol.x    
        actual_sol = interp_back( x )       
        back_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )
 
        
        sol , x = nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f , method='trapz' )       
        appr_sol = sol.x    
        actual_sol = interp_trapz( x )       
        trapz_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )
      
        sol , x = nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f , method='milne' )       
        appr_sol = sol.x    
        actual_sol = interp_milne( x )       
        milne_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )

        sol , x = nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f , method='AB' )       
        appr_sol = sol.x    
        actual_sol = interp_AB( x )       
        AB_err.append( np.max( np.abs( appr_sol - actual_sol ) ) )
        

    plt.figure()
    plt.grid(True)
    logx = np.log(  1/dims )
    logy = np.log( euler_err )
    
    coeffs = np.polyfit( logx , logy ,deg=1 )
    plt.loglog(1/dims, euler_err , linewidth=1, marker='v',
               markersize=10 , label='Euler\n conv. rate= '+str( round(coeffs[0],2)) )
    
    logx = np.log(  1/dims )
    logy = np.log( back_err )
    
    coeffs = np.polyfit( logx , logy ,deg=1 )
    plt.loglog(1/dims, back_err , linewidth=1, marker='*',
               markersize=10 , label='Backward\'s Euler\nconv. rate= '+str( round(coeffs[0],2)) )
    
    logx = np.log(  1/dims )
    logy = np.log( AB_err )
    
    coeffs = np.polyfit( logx , logy ,deg=1 )
    plt.loglog(1/dims, AB_err , linewidth=1, marker='s',
               markersize=10 , label='Adams-Bashforth\nconv. rate= '+str( round(coeffs[0],2)) )


    logx = np.log(  1/dims )
    logy = np.log( trapz_err )
    
    coeffs = np.polyfit( logx , logy ,deg=1 )
    plt.loglog(1/dims, trapz_err , linewidth=1, marker='o',
               markersize=10 , label='Trapezoidal\nconv. rate= '+str( round(coeffs[0],2)) )


    logx = np.log(  1/dims )
    logy = np.log( milne_err )
    
    coeffs = np.polyfit( logx , logy ,deg=1 )
    plt.loglog(1/dims, milne_err , linewidth=1, marker='s',
               markersize=10 , label='Milne\'s\nconv. rate= '+str( round(coeffs[0],2)) )



    #myaxes = list( plt.axis() )
    #myaxes[0]-=5
    #myaxes[1]+=5
    #plt.axis( myaxes )
    
    
    plt.xlabel('Grid size ($\Delta x$)' )
    plt.ylabel( '$\Vert u_*^{200} - u_*^N  \Vert_{\infty}$' )
    #plt.legend()
    plt.legend(bbox_to_anchor=(1.4, 0.9), bbox_transform=plt.gcf().transFigure , 
               fancybox=True, frameon=True)
    
    plt.savefig( '../images/FD_methods_comparison.png' , 
                bbox_inches='tight' , dpi=400, facecolor='white')


    plt.figure()
    sol , x = nonlinear_root( 200 , C_g , C_mu, C_a=C_a, C_f=C_f , method='trapz' )  
    
    plt.plot(x, sol.x , linewidth=2 )    
    plt.savefig( '../images/FD_example_solution.png' ,
                bbox_inches='tight' , dpi=400, facecolor='white')
    end = time.time()
    
    
    print "Elapsed time", round( end - start   , 2 ) ,  "seconds "


