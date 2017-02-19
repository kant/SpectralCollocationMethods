# -*- coding: utf-8 -*-
#Created on Oct 4, 2016
#@author: Inom Mirzaev

from __future__ import division

import numpy as np, matplotlib.pyplot as plt, seaborn as sns
import simpsons_rule as sr
import time

from scipy.integrate import quad, odeint, simps, trapz , cumtrapz
from mpl_toolkits.mplot3d import Axes3D

start = time.time()



fig_params = {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"],
        'axes.facecolor':'white' , 
        'figure.facecolor':'white' }
        
sns.set( context = 'paper' , style='white', palette='deep' , font='serif' , 
        font_scale=1,  rc=fig_params)


N=40

Cg_values = np.linspace(0.05, 1, 10 )
G_values = np.linspace( 1 , 10, 10 )
Cq_values  = np.zeros( [ len(Cg_values) , len(G_values) ] )

for mm in xrange( len( Cg_values ) ):
    for nn in xrange( len( G_values ) ):
        
        C_g = Cg_values[mm]
        
        G = G_values[nn]
        
        C_a = 1.3 * G    
        C_f = G  
        C_mu =np.exp(-G) 
        
        sol, x = sr.nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f )
        Cq_values[ mm , nn ] = 1 / simps( sr.renewal(x)*sol.x , x)        
        
from scipy import interpolate
        
myfunc = interpolate.interp2d( Cg_values , G_values , Cq_values ,  kind='cubic' )
Cg_values = np.linspace( np.min(Cg_values) , np.max(Cg_values) , 100 )
G_values = np.linspace( np.min(G_values) , np.max(G_values) , 100 )



Cq_values = myfunc(Cg_values, G_values)

X, Y = np.meshgrid(Cg_values, G_values) 
plt.close('all')
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

C_g = 1
G = 10
C_a = 1.3 * G    
C_f = G
  
C_mu =np.exp(-G) 

sol, x = sr.nonlinear_root( N , C_g , C_mu, C_a=C_a, C_f=C_f )

C_q = 1 / simps( sr.renewal(x)*sol.x , x)


ax.plot_surface(X, Y, Cq_values,          # data values (2D Arryas)
                       rstride=5,           # row step size
                       cstride=5,           # column step size
                       cmap='Blues',        # colour map
                       linewidth=0.1,       # wireframe line width
                       antialiased=True)
                       

ax.scatter(C_g, G, C_q , marker='*', s=100, c='red')
                

ax.set_xlabel('$C_g$')
ax.set_ylabel('$\dot{\gamma}$' )                       
ax.set_zlabel('$C_q$')

ax.view_init( azim=145 , elev=20 )
fig.savefig( '../images/renewal_rate_surface.png'  , dpi=400, facecolor='white')
#fig.show()





sns.set( context = 'paper' , style='white', palette='deep' , font='serif' , 
        font_scale=2,  rc=fig_params)

plt.figure()


Cq_values = [C_q , C_q-2, C_q+2] 
colors = ['red' , 'blue' , 'green']


for (C_q, color) in zip(Cq_values, colors):
    sol, x = sr.renewal_root(N, C_g=C_g , C_mu=C_mu, C_q=C_q,  C_a=C_a, C_f=C_f )
    plt.plot(x, sol.x , linewidth=2, color=color,
             label='$C_q=$'+str(round(C_q, 1) ) )    

plt.legend()

plt.savefig( '../images/renewal_rates.png'  , dpi=400, 
            bbox_inches='tight' , facecolor='white')
    
"""
#==============================================================================
# Renewal rate existence region
#==============================================================================
N=30
Cg_values = np.linspace( np.min(Cg_values) , np.max(Cg_values) , 20 )
G_values = np.linspace( np.min(G_values) , np.max(G_values) , 20 )
Cq_values = np.linspace( np.min(Cq_values) , np.max(Cq_values) , 20 )


region = np.zeros( [ len(Cg_values) , len(G_values) , len(Cq_values) ] )



count=0
for mm in xrange( len(Cg_values) ):
    C_g = Cg_values[mm]
    for nn in xrange(len(G_values)):
        G = G_values[nn]
        C_a = 1.3 * G    
        C_f = G  
        C_mu =np.exp(-G)
        
        for kk in xrange(len(Cq_values)):
            C_q = Cq_values[kk]
            try:
                sol , x  = sr.renewal_root(N , C_g=C_g , C_q=C_q, 
                                           C_a=C_a, C_mu=C_mu, C_f=C_f)            
            except:
                sol.success=False
                count +=1
                pass
            
            if sol.success==True and np.all(sol.x>0):
                region[mm, nn, kk] = 1
                
            else:
                region[mm, nn, kk] = np.NaN

                
fig = plt.figure()
ax = fig.gca(projection='3d') 
from scipy.spatial import  ConvexHull
from scipy.interpolate import griddata


grid_x, grid_y, grid_z = np.meshgrid(Cg_values, G_values, Cq_values, indexing='ij')

points = np.array([np.ravel(grid_x) , np.ravel(grid_y)  , np.ravel(grid_z) ] ).T
values = np.ravel(region)

  
exist = griddata( points , values , ( grid_x , grid_y , grid_z ) )
    
out = np.array([np.ravel(grid_x) , np.ravel(grid_y)  , np.ravel(grid_z) , np.ravel(exist)] ).T

mypts = out[ np.nonzero( np.isnan(out[:, 3] )==False )[0] ]
            
hull = ConvexHull( mypts[ : , 0:3] )
simp = hull.points[ hull.vertices ]

ax.plot_trisurf(mypts[:, 0] , mypts[:, 1] , mypts[:, 2] , triangles=hull.simplices, 
                linewidth=0.1, color='#8A2BE2', shade=False)
         

ax.set_xlabel('$C_g$')
ax.set_ylabel('$\dot{\gamma}$')                       
ax.set_zlabel('$C_q$')

ax.view_init( azim=145 , elev=20 )

                 
fig.savefig( '../images/renewal_rate_region.png'  , dpi=400, facecolor='white')
fig.show()   """    




        
end = time.time()


print "Elapsed time", round( end - start   , 2 ) ,  "seconds "


