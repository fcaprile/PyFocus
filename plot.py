# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:18:37 2020

@author: ferchi
"""
import numpy as np
from matplotlib import pyplot as plt
from plot_polarization_elipses import polarization_elipse

def plot_XZ_XY(ex1,ey1,ez1,ex2,ey2,ez2,zsteps,rsteps,field_of_view_x, field_of_view_z,n,wavelength,I_0,alpha,figure_name):
    
    zmax=field_of_view_z/2
    rmax=field_of_view_x*2**0.5/2
    
    x,y=np.shape(ex1)
    Ifield_xz=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            Ifield_xz[i,j]=np.real(ex1[i,j]*np.conj(ex1[i,j]) + ey1[i,j]*np.conj(ey1[i,j]) +ez1[i,j]*np.conj(ez1[i,j]))


    plt.rcParams['font.size']=14
    #intensity plot
    fig = plt.figure(num=str(figure_name)+': Intensity',figsize=(16, 8))
    spec = fig.add_gridspec(ncols=3, nrows=2)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.set_title('Intensity on xz')
    pos=ax1.imshow(Ifield_xz,extent=[-rmax,rmax,-zmax,zmax], interpolation='none', aspect='equal')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('z (nm)')
    cbar1= fig.colorbar(pos, ax=ax1, shrink=0.75)
    cbar1.ax.set_ylabel('Intensity (kW/cm\u00b2)')

    
    x2,y2=np.shape(ex2)
    Ifield_xy=np.zeros((x2,y2))
    for i in range(x2):
        for j in range(y2):
            Ifield_xy[i,j]=np.real(ex2[i,j]*np.conj(ex2[i,j]) + ey2[i,j]*np.conj(ey2[i,j]) +ez2[i,j]*np.conj(ez2[i,j]))

    xmax=field_of_view_x/2
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
    
    ax4 = fig.add_subplot(spec[1, 1])
    ax4.set_title('Polarization on xy')
    pos4=ax4.imshow(Ifield_xy,extent=[-xmax,xmax,-xmax,xmax],interpolation='none', aspect='auto',alpha=0.5)
    cbar4=fig.colorbar(pos4, ax=ax4)
    cbar4.ax.set_ylabel('Intensidad (kW/cm\u00b2)')
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('y (nm)')  
    ax4.axis('square')
    x_pos=np.linspace(-xmax*0.95,xmax*0.95,10)
    y_pos=np.linspace(-xmax*0.95,xmax*0.95,10)
    x_values=np.linspace(-xmax,xmax,np.shape(ex2)[0])
    y_values=np.linspace(xmax,-xmax,np.shape(ex2)[0])
    AMP=np.abs(xmax/6)
    for x_coor in x_pos:
        for y_coor in y_pos:
            x_index = (np.abs(x_values - x_coor)).argmin()
            y_index = (np.abs(y_values - y_coor)).argmin()
            if np.abs(Ifield_xy[y_index,x_index]/np.max(Ifield_xy))>0.15 and np.abs(ez2[y_index,x_index])**2/np.max(Ifield_xy)<0.9:
                if x_coor==x_pos[5] and y_coor==y_pos[5]: #at x,y=0,0 usually the atan2 does not work well
                    x_index+=1
                    y_index+=1
                polarization_elipse(ax4,x_coor,y_coor,ex2[y_index,x_index],ey2[y_index,x_index],AMP)
    '''
    #formula to calculate fwhm
    fwhm=np.abs(2*x_values[np.where(np.abs(Ifield_axis-np.max(Ifield_axis)/2)<0.05*np.max(Ifield_xy))[0]])
    print(fwhm)
    '''
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    #Amplitud de  and fase plot 
    Amp_max=np.abs(np.max([np.max(np.abs(ex2)),np.max(np.abs(ey2)),np.max(np.abs(ez2))]))**2
    plt.rcParams['font.size']=14
    #ex
    fig2, ((ax_x1,ax_y1,ax_z1),(ax_x2,ax_y2,ax_z2)) = plt.subplots(figsize=(15, 8),nrows=2, ncols=3)
    ax_x1.set_title('$|E_{f_x}|^2$')
    pos_x1=ax_x1.imshow(np.abs(ex2)**2/Amp_max,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_x1.set_xlabel('x (nm)')
    ax_x1.set_ylabel('y (nm)')
    ax_x1.axis('square')
    cbar_1_1=fig2.colorbar(pos_x1, ax=ax_x1)
    cbar_1_1.ax.set_ylabel('Relative intensity')
    
    ax_x2.set_title('$E_{f_x}$ phase')
    pos_x2=ax_x2.imshow(np.angle(ex2, deg=True)+180,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_x2.set_xlabel('x (nm)')
    ax_x2.set_ylabel('y (nm)')
    ax_x2.axis('square')
    cbar_1_1=fig2.colorbar(pos_x2, ax=ax_x2,ticks=[5, 90,180,270,355])
    cbar_1_1.ax.set_ylabel('Phase (º)')

    #ey
    ax_y1.set_title('$|E_{f_y}|^2$')
    pos_y1=ax_y1.imshow(np.abs(ey2)**2/Amp_max,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_y1.set_xlabel('x (nm)')
    ax_y1.set_ylabel('y (nm)')
    ax_y1.axis('square')
    cbar_1_1=fig2.colorbar(pos_y1, ax=ax_y1)
    cbar_1_1.ax.set_ylabel('Relative intensity')
    
    ax_y2.set_title('$E_{f_y}$ phase')
    pos_y2=ax_y2.imshow(np.angle(ey2, deg=True)+180,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_y2.set_xlabel('x (nm)')
    ax_y2.set_ylabel('y (nm)')
    ax_y2.axis('square')
    cbar_1_1=fig2.colorbar(pos_y2, ax=ax_y2,ticks=[5, 90,180,270,355])
    cbar_1_1.ax.set_ylabel('Phase (º)')

    #ez
    ax_z1.set_title('$|E_{f_z}|^2$')
    pos_z1=ax_z1.imshow(np.abs(ez2)**2/Amp_max,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_z1.set_xlabel('x (nm)')
    ax_z1.set_ylabel('y (nm)')
    ax_z1.axis('square')
    cbar_1_1=fig2.colorbar(pos_z1, ax=ax_z1)
    cbar_1_1.ax.set_ylabel('Relative intensity')
    
    ax_z2.set_title('$E_{f_z}$ phase')
    pos_z2=ax_z2.imshow(np.angle(ez2, deg=True)+180,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_z2.set_xlabel('x (nm)')
    ax_z2.set_ylabel('y (nm)')
    ax_z2.axis('square')
    cbar_1_1=fig2.colorbar(pos_z2, ax=ax_z2,ticks=[5, 90,180,270,355])
    cbar_1_1.ax.set_ylabel('Phase (º)')
    fig2.tight_layout()

def plot_XY(ex2,ey2,ez2,rsteps,field_of_view,n,wavelength,I_0,alpha,f,figure_name):   
    plt.rcParams['font.size']=14
    #intensity plot
    fig, (ax2, ax3) = plt.subplots(num=str(figure_name)+': Intensity',figsize=(12, 4), ncols=2)

    x2,y2=np.shape(ex2)
    Ifield_xy=np.zeros((x2,y2))
    for i in range(x2):
        for j in range(y2):
            Ifield_xy[i,j]=np.real(ex2[i,j]*np.conj(ex2[i,j]) + ey2[i,j]*np.conj(ey2[i,j]) +ez2[i,j]*np.conj(ez2[i,j]))

    xmax=field_of_view/2
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
    ax3.set_title('Intensity along x')
    ax3.plot(axis,Ifield_axis)
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('Intensity  (kW/cm\u00b2)')  
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)

    #amplitude and phase plot 
    Amp_max=np.abs(np.max([np.max(np.abs(ex2)),np.max(np.abs(ey2)),np.max(np.abs(ez2))]))
    #ex
    fig2, ((ax_x1,ax_y1,ax_z1),(ax_x2,ax_y2,ax_z2)) = plt.subplots(num=str(figure_name)+': Amplitudes',figsize=(18, 8),nrows=2, ncols=3)
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


