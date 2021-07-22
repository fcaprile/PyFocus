# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:22:34 2020

@author: ferchi

This script is meant to plot the polarization near the center of the donut produced by a VPP mask after inciding with right polarization
"""
import numpy as np
import os
from matplotlib import pyplot as plt
import time

from VPP import *
from plot import *
from plot_polarization_elipses import polarization_elipse

#all distance units are in mm
radius_VPP=5                 #PP radius (mm)
L=20000             #distance between mask and lens(mm)
I0=1                #intensity (kw/cm**2) 
Lambda=0.00075      #wavelength (mm)
FOV=2.2*radius_VPP           #field of view in wich to view the simulation of the field at the objective(mm)
laser_width=50             #gaussian beam's diameter

NA=1.4              #numeric aperture
n=1.5               #medium density
alpha=np.arcsin(NA/n)
f=radius_VPP/np.tan(alpha)*10**6   #focal distance, geometrical relation
zp0=0               #axial distance from focal plane
field_of_view=500
#polarization parameters:
psi=0               #(degrees) amplitude relation between Ex and Ey, Ex/Ey=tan(psi)
delta=0             #(degrees) initial phase of Ey for eliptical or circular polarization: +90 for left circular, -90 for right circular
wavelength=750
zsteps=75
rsteps=2
phip0=0

II1,II2,II3,II4,II5=VPP_integration(alpha,n,f,radius_VPP,wavelength,zsteps,rsteps,field_of_view,laser_width)
fields_VPP=VPP_fields(II1,II2,II3,II4,II5,wavelength,I0,psi,delta,zsteps,rsteps,field_of_view,phip0,n,f,zp0)
time.sleep(0.5)
ex1,ey1,ez1,ex2,ey2,ez2=np.array(fields_VPP)

zmax=np.int(field_of_view/2)
rmax=np.int(np.rint(field_of_view*2**0.5/2))

x,y=np.shape(ex1)
Ifield_xz=np.zeros((x,y))
for i in range(x):
    for j in range(y):
        Ifield_xz[i,j]=np.real(ex1[i,j]*np.conj(ex1[i,j]) + ey1[i,j]*np.conj(ey1[i,j]) +ez1[i,j]*np.conj(ez1[i,j]))


plt.rcParams['font.size']=14
#intensity plot
fig = plt.figure(figsize=(16, 4))
spec = fig.add_gridspec(ncols=3, nrows=1)
fig.suptitle('Field at focal plane')
ax1 = fig.add_subplot(spec[0, 0])
ax1.set_title('Intensity on xz')
pos=ax1.imshow(Ifield_xz,extent=[-rmax,rmax,-zmax,zmax], interpolation='none', aspect='auto')
ax1.set_xlabel('x (nm)')
ax1.set_ylabel('z (nm)')
cbar1= fig.colorbar(pos, ax=ax1)
cbar1.ax.set_ylabel('Intensity (kW/cm\u00b2)')


x2,y2=np.shape(ex2)
Ifield_xy=np.zeros((x2,y2))
for i in range(x2):
    for j in range(y2):
        Ifield_xy[i,j]=np.real(ex2[i,j]*np.conj(ex2[i,j]) + ey2[i,j]*np.conj(ey2[i,j]) +ez2[i,j]*np.conj(ez2[i,j]))

# xy_res=field_of_view/np.int(np.rint(field_of_view/rsteps)-2)
# Int_final=np.sum(Ifield_xy)*(xy_res)**2
# transmission=Int_final/(np.pi*(np.tan(alpha)*f)**2*I_0)        
# print('Transmission= ',transmission)                

xmax=field_of_view/2
ax2 = fig.add_subplot(spec[0, 1])
ax2.set_title('Intensity on xy')
pos2=ax2.imshow(Ifield_xy,extent=[-xmax,xmax,-xmax,xmax],interpolation='none', aspect='auto')
cbar2=fig.colorbar(pos2, ax=ax2)
ax2.set_xlabel('x (nm)')
ax2.set_ylabel('y (nm)')  
ax2.axis('square')
cbar2.ax.set_ylabel('Intensity (kW/cm\u00b2)')

x2=np.shape(Ifield_xy)[0]
Ifield_axis=Ifield_xy[int(x2/2),:]
axis=np.linspace(-xmax,xmax,x2)
ax3 = fig.add_subplot(spec[0, 2])
ax3.set_title('Intensity along x')
ax3.plot(axis,Ifield_axis)
ax3.set_xlabel('x (nm)')
ax3.set_ylabel('Intensity  (kW/cm\u00b2)')  
fig.tight_layout()
fig.subplots_adjust(top=0.90)

fig3 = plt.figure(figsize=(16, 4))
spec = fig3.add_gridspec(ncols=3, nrows=1)
ax4 = fig3.add_subplot(spec[0, 0])
ax4.set_title('Polarization on xy')
pos4=ax4.imshow(Ifield_xy,extent=[-xmax,xmax,-xmax,xmax],interpolation='none', aspect='auto',alpha=0.5)
fig.colorbar(pos4, ax=ax4)
ax4.set_xlabel('x (nm)')
ax4.set_ylabel('y (nm)')  
ax4.axis('square')
x_pos=np.linspace(-xmax*0.95,xmax*0.95,11)
y_pos=np.linspace(-xmax*0.95,xmax*0.95,11)
x_values=np.linspace(-xmax,xmax,np.shape(ex2)[0])
y_values=np.linspace(xmax,-xmax,np.shape(ex2)[0])
AMP=np.abs(xmax/6)
for x_coor in x_pos:
    for y_coor in y_pos:
        x_index = (np.abs(x_values - x_coor)).argmin()
        y_index = (np.abs(y_values - y_coor)).argmin()
        polarization_elipse(ax4,x_coor,y_coor,ex2[y_index,x_index],ey2[y_index,x_index],AMP)
            
ax5 = fig3.add_subplot(spec[0, 1])
ax5.set_title('tan-1(Ey/Ex)')
# phies=np.zeros((x2,y2))
phies=np.arctan2(np.abs(ey2),np.abs(ex2))/np.pi*180
phies[int(x2/2),int(y2/2)]=45
pos=ax5.imshow(phies,extent=[-xmax,xmax,-xmax,xmax],interpolation='none', aspect='auto')
cbar=fig3.colorbar(pos, ax=ax5)
ax5.set_xlabel('x (nm)')
ax5.set_ylabel('y (nm)')  
ax5.axis('square')
cbar.ax.set_ylabel('Degrees')

betas=np.zeros((x2,y2))
for i in range(x2):
    for j in range(y2):
        beta=(np.angle(ey2[i,j])-np.angle(ex2[i,j]))
        if beta>np.pi:
            beta-=2*np.pi
        if beta<-np.pi:
            beta+=2*np.pi        
        betas[i,j]=beta/np.pi*180
betas[int(x2/2),int(y2/2)]=90
            
            
ax6 = fig3.add_subplot(spec[0, 2])
ax6.set_title('\u0394 phase')
pos=ax6.imshow(betas,extent=[-xmax,xmax,-xmax,xmax],interpolation='none', aspect='auto')
cbar=fig3.colorbar(pos, ax=ax6)
ax6.set_xlabel('x (nm)')
ax6.set_ylabel('y (nm)')  
ax6.axis('square')
cbar.ax.set_ylabel('Degrees')
fig3.tight_layout()
fig3.subplots_adjust(top=0.90)


#amplitude and phase plot 
Amp_max=np.abs(np.max([np.max(np.abs(ex2)),np.max(np.abs(ey2)),np.max(np.abs(ez2))]))
#ex
fig2, ((ax_x1,ax_y1,ax_z1),(ax_x2,ax_y2,ax_z2)) = plt.subplots(figsize=(18, 8),nrows=2, ncols=3)
fig2.suptitle('Field at focal plane')
ax_x1.set_title('Ex amplitude')
pos_x1=ax_x1.imshow(np.abs(ex2)/Amp_max,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
ax_x1.set_xlabel('x (nm)')
ax_x1.set_ylabel('y (nm)')
ax_x1.axis('square')
cbar_1_1=fig2.colorbar(pos_x1, ax=ax_x1)
cbar_1_1.ax.set_ylabel('Relative amplitude')

ax_x2.set_title('Ex phase')
pos_x2=ax_x2.imshow(np.angle(ex2, deg=True)+180,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
ax_x2.set_xlabel('x (nm)')
ax_x2.set_ylabel('y (nm)')
ax_x2.axis('square')
cbar_1_1=fig2.colorbar(pos_x2, ax=ax_x2,ticks=[5, 90,180,270,355])
cbar_1_1.ax.set_ylabel('Angle (Degrees)')

#ey
ax_y1.set_title('Ey amplitude')
pos_y1=ax_y1.imshow(np.abs(ey2)/Amp_max,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
ax_y1.set_xlabel('x (nm)')
ax_y1.set_ylabel('y (nm)')
ax_y1.axis('square')
cbar_1_1=fig2.colorbar(pos_y1, ax=ax_y1)
cbar_1_1.ax.set_ylabel('Relative amplitude')

ax_y2.set_title('Ey phase')
pos_y2=ax_y2.imshow(np.angle(ey2, deg=True)+180,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
ax_y2.set_xlabel('x (nm)')
ax_y2.set_ylabel('y (nm)')
ax_y2.axis('square')
cbar_1_1=fig2.colorbar(pos_y2, ax=ax_y2,ticks=[5, 90,180,270,355])
cbar_1_1.ax.set_ylabel('Angle (Degrees)')

#ez
ax_z1.set_title('Ez amplitude')
pos_z1=ax_z1.imshow(np.abs(ez2)/Amp_max,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
ax_z1.set_xlabel('x (nm)')
ax_z1.set_ylabel('y (nm)')
ax_z1.axis('square')
cbar_1_1=fig2.colorbar(pos_z1, ax=ax_z1)
cbar_1_1.ax.set_ylabel('Relative amplitude')

ax_z2.set_title('Ez phase')
pos_z2=ax_z2.imshow(np.angle(ez2, deg=True)+180,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
ax_z2.set_xlabel('x (nm)')
ax_z2.set_ylabel('y (nm)')
ax_z2.axis('square')
cbar_1_1=fig2.colorbar(pos_z2, ax=ax_z2,ticks=[5, 90,180,270,355])
cbar_1_1.ax.set_ylabel('Angle (Degrees)')
fig2.tight_layout()
fig2.subplots_adjust(top=0.88)


