# -*- coding: utf-8 -*-
#Created on Oct 4, 2016
#@author: Inom Mirzaev


from simpsons_rule import *

    
start = time.time()

C_g = 0.1

G = 5
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
#   Convergence to a non-trivial steady state solution
#==============================================================================

N=30

sim_time = np.linspace(0, 10, 200)
#y0 = 2*IC(x)

#y0  = np.ones_like(x)

sol , x = nonlinear_root( N=N , C_f=C_f , C_mu=C_mu  , 
                             C_g = C_g , C_a = C_a )

y0 = 2*np.exp(-10*x)
    
C_q = 1 / simps( renewal(x)*sol.x , x)  



yout, sim_time , x = simulate_ode( sim_time , y0 ,  C_g=C_g , C_mu=C_mu , 
                                  C_q = C_q , C_a = C_a , C_f=C_f , N=N )


X, Y = np.meshgrid(x[1:], sim_time) 

plt.close('all')
fig = plt.figure()
ax = fig.gca(projection='3d')               # 3d axes instance
surf = ax.plot_surface(X, Y, yout[:, 1:],          # data values (2D Arryas)
                       rstride=5,           # row step size
                       cstride=5,           # column step size
                       cmap='Blues',        # colour map
                       linewidth=0.1,       # wireframe line width
                       antialiased=True)
                       

ax.set_xlabel('$x$', fontsize=20)
ax.set_ylabel('$t$', fontsize=20)                       
ax.set_zlabel('$u(t,x)$', fontsize=20)

ax.view_init( azim=-40 , elev=30 )
fig.savefig( '../images/solutions_convergence.png'  , dpi=400, facecolor='white')



#==============================================================================
#   Convergence to zero steady state solution
#==============================================================================

C_q = 1
y0  = np.ones_like(x)

yout, sim_time , x = simulate_ode( sim_time , y0 ,  C_g=C_g , C_mu=C_mu , 
                                  C_q = C_q , C_a = C_a , C_f=C_f , N=N )


X, Y = np.meshgrid(x[1:], sim_time) 

fig = plt.figure()
ax = fig.gca(projection='3d')               # 3d axes instance
surf = ax.plot_surface(X, Y, yout[:, 1:],          # data values (2D Arryas)
                       rstride=5,           # row step size
                       cstride=5,           # column step size
                       cmap='Blues',        # colour map
                       linewidth=0.1,       # wireframe line width
                       antialiased=True)
                       

ax.set_xlabel('$x$', fontsize=20)
ax.set_ylabel('$t$', fontsize=20)                       
ax.set_zlabel('$u(t,x)$', fontsize=20)

ax.view_init( azim=-40 , elev=30 )
fig.savefig( '../images/solutions_conv_zero.png'  , dpi=400, facecolor='white')



#==============================================================================
#   Divergence
#==============================================================================

C_q = 10
C_g = 0.01

y0 = np.exp(-10*x)
#y0 = 2*IC(x)
#y0 = np.exp(-10*x)
#y0  = 2*np.ones_like(x)
yout, sim_time , x = simulate_ode( sim_time , y0 ,  C_g=C_g , C_mu=C_mu , 
                                  C_q = C_q , C_a = C_a , C_f=C_f , N=N )


X, Y = np.meshgrid(x[1:], sim_time) 

fig = plt.figure()
ax = fig.gca(projection='3d')               # 3d axes instance
surf = ax.plot_surface(X, Y, yout[:, 1:],          # data values (2D Arryas)
                       rstride=5,           # row step size
                       cstride=5,           # column step size
                       cmap='Blues',        # colour map
                       linewidth=0.1,       # wireframe line width
                       antialiased=True)
                       

ax.set_xlabel('$x$', fontsize=20)
ax.set_ylabel('$t$', fontsize=20)                       
ax.set_zlabel('$u(t,x)$', fontsize=20)

ax.view_init( azim=-40 , elev=30 )
fig.savefig( '../images/solutions_divergence.png'  , dpi=400, facecolor='white')
#plt.show()

end = time.time()


print "Elapsed time", round( end - start   , 2 ) ,  "seconds "


