import user_interface
import sim
import plot
import numpy as np

# ui=user_interface.UI()
# ui.show()
NA=1.4
n=1.5
h=3
w0=5
wavelength=640
gamma=45
beta=90
z=0
x_steps=30
z_steps=60
x_range=800
z_range=1500
L=''
R=''
ds=''
I0=1
z_int=''
figure_name='Ejemplo'
parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)#VPP mask:

# n=np.array((1.5,1.33))
# ds=np.array((np.inf,np.inf))
# z_int=0
# parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)#VPP mask:

# fields=sim.no_mask(True,False,*parameters)

# fields=sim.VP(False,False,*parameters)

L=10000
R=5
parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)#VPP mask:
# fields=sim.VP(True,False,*parameters)
#custom simulation

#VP
# mask_function= lambda rho,phi,w0,f,k: np.exp(1j*phi)#VP

#no mask
# mask_function= lambda rho,phi,w0,f,k: 1

#displacement without mask
# d=0.5
# rho2=lambda rho,phi:(rho**2+d**2-2*rho*d*np.cos(phi))**0.5
# phi2=lambda rho,phi:np.arctan2(rho*np.sin(phi),rho*np.cos(phi)-d)
# mask_function=lambda rho,phi,w0,f,k: np.exp(-(rho2(rho,phi)/w0)**2)#displaced a distance d

#displacement with VP mask
d=0.4*h
rho2=lambda rho,phi:(rho**2+d**2-2*rho*d*np.cos(phi))**0.5
phi2=lambda rho,phi:np.arctan2(rho*np.sin(phi),rho*np.cos(phi)-d)
mask_function=lambda rho,phi,w0,f,k: np.exp(-(rho2(rho,phi)/w0)**2+1j*phi2(rho,phi))#displaced a distance d with a VPP mask

#tilt without displacement
# ang=3.11*10**-5*10
# mask_function=lambda rho,phi,w0,f,k: np.exp(-(rho/w0)**2+1j*(phi+k*np.sin(ang)*rho*np.cos(phi)))#displaced a distance d with a VPP mask

# tilt with displacement
# ang=3.11*10**-5
# f=h*n/NA
# d=f*np.tan(ang)
# rho2=lambda rho,phi:(rho**2+d**2-2*rho*d*np.cos(phi))**0.5
# phi2=lambda rho,phi:np.arctan2(rho*np.sin(phi),rho*np.cos(phi)-d)
# mask_function=lambda rho,phi,w0,f,k: np.exp(-(rho2(rho,phi)/w0)**2+1j*(phi2(rho,phi)+k*np.sin(ang)*rho*np.cos(phi2(rho,phi))))#displaced a distance d with a VPP mask


#for TIRF:
# def mask_function(rho,phi,w0,f,k):
#     theta_crit=np.arcsin(n[1]/n[0])
#     if rho>f*np.sin(theta_crit):
#         return np.exp(1j*phi)
#     else:
#         return 0
fields=sim.custom(mask_function,True,False,*parameters,200,200)
# n=np.array((1.5,0.14+3.55j,1.54,1.33))
# ds= np.array((np.inf,44,24,np.inf))
# z_int=-0
# parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)#VPP mask:
# fields=sim.VP(propagation=False,interface=True,*parameters)
plot.plot_XZ_XY(*fields,x_range,z_range,figure_name)

