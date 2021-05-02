# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 22:06:18 2020

@author: Fer Caprile
"""
from trapezoid2D_integration_functions import *
import numpy as np
import time
from matplotlib import pyplot as plt
from plot import *
#%%
'''
tipical simulation of the field at the focal plane by asuming the incident field to be a gaussian beam:
    
all distance units are in mm
'''
h=5                     #radius of aperture (mm)
L=1000                  #distance between mask and lens(mm)
I0=1                    #intensity (kw/cm**2) 
Lambda=0.00075          #wavelength (mm)
FOV=2.2*h               #field of view in wich to view the simulation of the field at the objective(mm)
radius=10                 #gaussian beam's radius

NA=1.4                  #numeric aperture
n=1.5                   #medium density
alpha=np.arcsin(NA/n)
focus=h/np.sin(alpha)   #focal distance given by sine's condition
zp0=0                   #axial distance from focal plane

#polarization parameters:
psi=45                  #(degrees) amplitude relation between Ex and Ey, Ex/Ey=tan(psi)
delta=90                #(degrees) initial phase of Ey for eliptical or circular polarization: +90 for left circular, -90 for right circular

#resolution and calculus precision parameters:
#at higher number of division better resolution
N_rho=400               #number of divisions in rho for the trapezoidal cuadrature for calculating field at the objective
N_phi=400               #number of divisions in phi for the trapezoidal cuadrature for calculating field at the objective

resolution_theta=200    #resolution in theta for the field at the objective
resolution_phi=100      #resolution in phi for the field at the objective
resolution_focus=60     #resolution for field at the focal plane XY
resolution_z=50         #resolution for field at the focal plane XZ
FOV_focus=1500          #field of view in wich to view the simulation of the fiald in the focal XY plane(nm)
z_FOV=3000              #field of view for the XZ plane (nm)
'''
Next define the mask function, the example for a VPP is given
The parameters: beam's radius (radius), focal length (focus) and wavenumber (k) are given as a parameter for the function in case they appear in the phase mask
'''
mask_function=lambda rho, phi,radius,focus,k: np.exp(1j*phi)   
  
#calculate field at the entrance of the lens:                    
ex_lens,ey_lens=generate_incident_field(mask_function,alpha,focus,resolution_phi,resolution_theta,h,psi,delta,radius,I0,Lambda)

#optional: if one wants to plot the field produced by the phase mask at the entrance of the lens:
#plot_in_cartesian(ex_lens,ey_lens,h,alpha,focus)
time.sleep(0.1)
#calculation of field at the focal plane:
Ex_XZ,Ey_XZ,Ez_XZ,Ex_XY,Ey_XY,Ez_XY=custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,Lambda,z_FOV,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,FOV_focus)    
#ploting the intensity and amplitudes
rsteps=FOV_focus/resolution_focus #size of each pixel in rho
zsteps=z_FOV/resolution_z         #size of each pixel in z
plot_XZ_XY(Ex_XZ,Ey_XZ,Ez_XZ,Ex_XY,Ey_XY,Ez_XY,zsteps,rsteps,FOV_focus,z_FOV,n,Lambda,I0,alpha)

#%%
'''
tipical simulation of the field at the entrance of the lens and at the focal plane:
    
all distance units are in mm
'''
h=5                     #radius of aperture (mm)
L=1000                  #distance between mask and lens(mm)
I0=1                    #intensity (kw/cm**2) 
Lambda=0.00075          #wavelength (mm)
FOV=2.2*h               #field of view in wich to view the simulation of the field at the objective(mm)
radius=50                 #gaussian beam's radius

NA=1.4                  #numeric aperture
n=1.5                   #medium density
alpha=np.arcsin(NA/n)
focus=h/np.sin(alpha)   #focal distance given by sine's condition
zp0=0                   #axial distance from focal plane

#polarization parameters:
psi=45                  #(degrees) amplitude relation between Ex and Ey, Ex/Ey=tan(psi)
delta=90                #(degrees) initial phase of Ey for eliptical or circular polarization: +90 for left circular, -90 for right circular

#resolution and calculus precision parameters:
#at higher number of division better resolution
N_rho=400               #number of divisions in rho for the trapezoidal cuadrature for calculating field at the objective
N_phi=400               #number of divisions in phi for the trapezoidal cuadrature for calculating field at the objective
#notice that tipical N_phi is set lower since most fields don't variate rapidly along phi
resolution_theta=200    #resolution in theta for the field at the objective
resolution_phi=100      #resolution in phi for the field at the objective
resolution_focus=40     #resolution for field at the focal plane XY
resolution_z=30         #resolution for field at the focal plane XZ
FOV_focus=1500          #field of view in wich to view the simulation of the fiald in the focal XY plane(nm)
z_FOV=3000              #field of view for the XZ plane (nm)
#VPP mask:
mask_function=lambda rho, phi: np.exp(1j*phi)   
#calculate field at the entrance of the lens:                    
ex_lens, ey_lens,I_cartesian,Ex_cartesian,Ey_cartesian=custom_mask_objective_field(psi,delta,resolution_theta,resolution_phi,N_rho,N_phi,alpha,mask_function,h,L,I0,Lambda,radius,plot=False)
#ex_lens and ey_lens are calculated in polar coordinates, I_cartesan is the intensity in cartesian coordinates, ex_cartesian and ey_cartesian are the X and Y components in cartesian

time.sleep(0.1)
#calculation of field at the focal plane:
Ex_XZ,Ey_XZ,Ez_XZ,Ex_XY,Ey_XY,Ez_XY=custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,Lambda,z_FOV,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,FOV_focus,x0)    
#ploting the intensity and amplitudes
rsteps=FOV_focus/resolution_focus #size of each pixel in rho
zsteps=z_FOV/resolution_z         #size of each pixel in z
plot_XZ_XY(Ex_XZ,Ey_XZ,Ez_XZ,Ex_XY,Ey_XY,Ez_XY,zsteps,rsteps,z_FOV,FOV_focus,n,Lambda,I0,alpha,focus)

#%%
'''
if one wants to only simulate the field at the objective:

all distance units are in mm
'''
h=5                 #radius of aperture (mm)
L=20000             #distance between mask and lens(mm)
I0=1                #intensity (kw/cm**2) 
Lambda=0.00075      #wavelength (mm)
FOV=2.2*h           #field of view in wich to view the simulation of the field at the objective(mm)
radius=50             #gaussian beam's radius

#polarization parameters:
psi=0               #(degrees) amplitude relation between Ex and Ey, Ex/Ey=tan(psi)
delta=0             #(degrees) initial phase of the y field for eliptical or circular polarization: +90 for left circular, -90 for right circular

#resolution and calculus precision parameters:
#at higher number of division better resolution
N_rho=400           #number of divisions in rho for the trapezoidal cuadrature for calculating field at the objective
N_phi=100           #number of divisions in phi for the trapezoidal cuadrature for calculating field at the objective

resolution_objective=100 #resolution for field at the objective

#VPP mask:
mask_function=lambda rho, phi: np.exp(1j*phi)  
#distorsion + VPP:
gamma=0.00005
mask_function=lambda rho, phi:np.exp(1j*(phi+k*rho*np.sin(gamma)*np.cos(phi)))  
ex_lens, ey_lens,I_cartesian,Ex,Ey=custom_mask_objective_field_GPU(psi,delta,resolution_theta,resolution_phi,N_rho,N_phi,alpha,mask_function,h,L,I0,Lambda,radius,plot=False)
#ex_lens and ey_lens are calculated in polar coordinates, I_cartesan is the intensity in cartesian coordinates, ex_cartesian and ey_cartesian are the X and Y components in cartesian

plt.rcParams['font.size']=20#tamaño de fuente
fig1, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)
fig1.suptitle('Field at the objective using 2D integration')
xmax=FOV/2
Ifield=np.abs(Ex)**2+np.abs(Ey)**2
ax1.set_title('Intensity')
pos=ax1.imshow(Ifield,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
ax1.set_xlabel('x (um)')
ax1.set_ylabel('y (um)')
ax1.axis('square')
cbar1= fig1.colorbar(pos, ax=ax1)
cbar1.ax.set_ylabel('Intensity (kW/cm\u00b2)')


x2=np.shape(Ifield)[0]
ax2.set_title(' Intensity along x')
ax2.plot(np.linspace(-xmax,xmax,x2),Ifield[int(x2/2),:])
ax2.set_xlabel('x (mm)')
ax2.set_ylabel('Intensity  (kW/cm\u00b2)')  
fig1.tight_layout()
fig1.subplots_adjust(top=0.80)
#%%
'''
to quickly test if the used resolution for the trapezoidal integration is enought, this function does the calculus for 1/20 of the polar anglu phi and the extrapolates the same value for other angles (not good for an assimetric masks)
it simulates the fields at the objective and at the focal plane:

all distance units are in mm
'''
h=5                 #radius of aperture (mm)
L=300             #distance between mask and lens(mm)
I0=1                #intensity (kw/cm**2) 
Lambda=0.00075      #wavelength (mm)
FOV=2.2*h           #field of view in wich to view the simulation of the field at the objective(mm)
radius=50             #gaussian beam's radius

NA=1.4              #numeric aperture
n=1.5               #medium density
alpha=np.arcsin(NA/n)
focus=h/np.sin(alpha)   #focal distance given by sine's condition
zp0=0               #axial distance from focal plane

#polarization parameters:
psi=0               #(degrees) amplitude relation between Ex and Ey, Ex/Ey=tan(psi)
delta=0             #(degrees) initial phase of Ey for eliptical or circular polarization: +90 for left circular, -90 for right circular

#resolution and calculus precision parameters:
#at higher number of division better resolution
N_rho=500          #number of divisions in rho for the trapezoidal cuadrature for calculating field at the objective
N_phi=500           #number of divisions in phi for the trapezoidal cuadrature for calculating field at the objective
#notice that tipical N_phi is set lower since most fields don't variate rapidly along phi
resolution_theta=200    #resolution in theta for the field at the objective
resolution_phi=100       #resolution in phi for the field at the objective
resolution_focus=40 #resolution for field at the focal plane
FOV_focus=1800      #field of view in wich to view the simulation of the fiald in the focal plane(mm)

#VPP mask:
mask_function=lambda rho, phi: np.exp(1j*phi)   #write using np in order to be used with the GPU
#distorsion + VPP:
gamma=0.00005
mask_function=lambda rho, phi:np.exp(1j*(phi+k*rho*np.sin(gamma)*np.cos(phi)))   #write using in order to be used with the GPU
#calculation of field at the objective:
test_custom_mask_objective_field(psi,delta,resolution_theta,resolution_phi,N_rho,N_phi,alpha,mask_function,h,L,I0,Lambda,radius,plot=True)
#ex_lens and ey_lens are calculated in polar coordinates, I_cartesan is the intensity in cartesian coordinates, ex_cartesian and ey_cartesian are the X and Y components in cartesian
