import numpy as np
from matplotlib import pyplot as plt
from auxiliary.plot_polarization_elipses import polarization_elipse

def plot_XZ_XY(ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY,x_range,z_range,figure_name=''):
    '''
    Plot the calculated fields ont the XY and XZ planes
    
    parameters are ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY, each one is a matrix with the amplitude of each cartesian component on the XZ plane (ex_XZ,ey_XZ,ez_XZ) or on the XY plane (ex_XY,ey_XY,ez_XY)
    
    Each index of the matrixes corresponds to a different pair of coordinates, for example: 
    ex_XZ[z,x] with z each index of the coordinates np.linspace(z_range/2,-z_range/2,2*int(z_range/zsteps/2)) and x each index for np.linspace(-x_range/2**0.5,x_range/2**0.5,2*int(x_range/rsteps/2**0.5)) in which the field is calculated
    ex_XZ[y,x2] with y each index of the coordinates np.linspace(x_range/2,-x_range/2,2*int(x_range/rsteps/2)) and x each index for np.linspace(-x_range/2,x_range/2,2*int(x_range/rsteps/2)) in which the field is calculated
    
    The intensity is ploted in (kW/cm^2) since most focused fields at NA=1.4 have a maximum intensity in the order of 10^6 to 10^8 mW/cm^2
    '''
    #For pasage from (mW/cm^2) to (kW/cm^2) the intensity will be divided by 10**6
    
    zmax=z_range/2
    rmax=x_range*2**0.5/2 #the maximum radial distance is calculated sqrt(2) times biger than the maximum x or y distnace in the previous functions
    x,y=np.shape(ex_XZ)

    radial_pixel_width=x_range*2**0.5/2/y#value used to show the pixels centered at the radial position at which they are calculated
    Ifield_xz=np.abs(ex_XZ)**2+np.abs(ey_XZ)**2+np.abs(ez_XZ)**2
    Ifield_xz/=10**6
    
    plt.rcParams['font.size']=14
    #intensity plot
    fig = plt.figure(num=str(figure_name)+'_Intensity',figsize=(16, 8))
    spec = fig.add_gridspec(ncols=3, nrows=2)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.set_title('Normalized intensity',pad=20)
    pos=ax1.imshow(Ifield_xz,extent=[-rmax-radial_pixel_width,rmax-radial_pixel_width,-zmax,zmax], interpolation='none', aspect='equal')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('z (nm)')
    cbar1= fig.colorbar(pos, ax=ax1)
    cbar1.ax.set_ylabel('Intensity (kW/cm\u00b2)')

    
    x2,y2=np.shape(ex_XY)
    Ifield_xy=np.abs(ex_XY)**2+np.abs(ey_XY)**2+np.abs(ez_XY)**2
    Ifield_xy/=10**6

    xmax=x_range/2
    extent=[-xmax-radial_pixel_width,xmax-radial_pixel_width,-xmax+radial_pixel_width,xmax+radial_pixel_width]
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_title('Intensity on xy',pad=20)
    pos2=ax2.imshow(Ifield_xy,extent=extent,interpolation='none', aspect='auto')
    cbar2=fig.colorbar(pos2, ax=ax2)
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')  
    ax2.axis('square')
    cbar2.ax.set_ylabel('Intensity (kW/cm\u00b2)')
    
    x2=np.shape(Ifield_xy)[0]
    Ifield_axis=Ifield_xy[int(x2/2),:]
    axis=np.linspace(-xmax-radial_pixel_width,xmax-radial_pixel_width,x2)
    ax3 = fig.add_subplot(spec[0, 2])
    ax3.set_title('Intensity along x',pad=20)
    ax3.plot(axis,Ifield_axis)
    ax3.grid(True)
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('Intensity  (kW/cm\u00b2)')  
    
    ax4 = fig.add_subplot(spec[1, 1])
    ax4.set_title('Polarization on xy')
    pos4=ax4.imshow(Ifield_xy,extent=extent,interpolation='none', aspect='auto',alpha=0.5)
    cbar4=fig.colorbar(pos4, ax=ax4)
    cbar4.ax.set_ylabel('Intensity (kW/cm\u00b2)')
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('y (nm)')  
    ax4.axis('square')
    x_pos=np.linspace(-xmax*0.95,xmax*0.95,10)
    y_pos=np.linspace(-xmax*0.95,xmax*0.95,10)
    x_values=np.linspace(-xmax,xmax,np.shape(ex_XY)[0])
    y_values=np.linspace(xmax,-xmax,np.shape(ex_XY)[0])
    AMP=np.abs(xmax/6)
    for x_coor in x_pos:
        for y_coor in y_pos:
            x_index = (np.abs(x_values - x_coor)).argmin()
            y_index = (np.abs(y_values - y_coor)).argmin()
            # if np.abs(Ifield_xy[y_index,x_index]/np.max(Ifield_xy))>0.15 and np.abs(ez_XY[y_index,x_index])**2/np.max(Ifield_xy)<0.9:#added a condition for the z intensity to avoid point where the polarization is mostly along z
            if np.abs(Ifield_xy[y_index,x_index]/np.max(Ifield_xy))>0.15: #removed the z intensity condition
                if x_coor==x_pos[5] and y_coor==y_pos[5]: #at x,y=0,0 usually the atan2 does not work well
                    x_index+=1
                    y_index+=1
                polarization_elipse(ax4,x_coor,y_coor,ex_XY[y_index,x_index],ey_XY[y_index,x_index],AMP)
    '''
    #formula to calculate fwhm
    fwhm=np.abs(2*x_values[np.where(np.abs(Ifield_axis-np.max(Ifield_axis)/2)<0.05*np.max(Ifield_xy))[0]])
    print(fwhm)
    '''
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    #Amplitud de  and fase plot 
    Amp_max=np.abs(np.max([np.max(np.abs(ex_XY)),np.max(np.abs(ey_XY)),np.max(np.abs(ez_XY))]))**2
    plt.rcParams['font.size']=14
    #ex
    fig2, ((ax_x1,ax_y1,ax_z1),(ax_x2,ax_y2,ax_z2)) = plt.subplots(num=str(figure_name)+'_Amplitude',figsize=(15, 8),nrows=2, ncols=3)
    ax_x1.set_title('$|E_{f_x}|^2$')
    pos_x1=ax_x1.imshow(np.abs(ex_XY)**2/Amp_max,extent=extent, interpolation='none', aspect='auto')
    ax_x1.set_xlabel('x (nm)')
    ax_x1.set_ylabel('y (nm)')
    ax_x1.axis('square')
    cbar_1_1=fig2.colorbar(pos_x1, ax=ax_x1)
    cbar_1_1.ax.set_ylabel('Relative intensity')
    
    ax_x2.set_title('$E_{f_x}$ phase')
    pos_x2=ax_x2.imshow(np.angle(ex_XY, deg=True)+180,extent=extent, interpolation='none', aspect='auto')
    ax_x2.set_xlabel('x (nm)')
    ax_x2.set_ylabel('y (nm)')
    ax_x2.axis('square')
    cbar_1_1=fig2.colorbar(pos_x2, ax=ax_x2,ticks=[5, 90,180,270,355])
    cbar_1_1.ax.set_ylabel('Phase (º)')

    #ey
    ax_y1.set_title('$|E_{f_y}|^2$')
    pos_y1=ax_y1.imshow(np.abs(ey_XY)**2/Amp_max,extent=extent, interpolation='none', aspect='auto')
    ax_y1.set_xlabel('x (nm)')
    ax_y1.set_ylabel('y (nm)')
    ax_y1.axis('square')
    cbar_1_1=fig2.colorbar(pos_y1, ax=ax_y1)
    cbar_1_1.ax.set_ylabel('Relative intensity')
    
    ax_y2.set_title('$E_{f_y}$ phase')
    pos_y2=ax_y2.imshow(np.angle(ey_XY, deg=True)+180,extent=extent, interpolation='none', aspect='auto')
    ax_y2.set_xlabel('x (nm)')
    ax_y2.set_ylabel('y (nm)')
    ax_y2.axis('square')
    cbar_1_1=fig2.colorbar(pos_y2, ax=ax_y2,ticks=[5, 90,180,270,355])
    cbar_1_1.ax.set_ylabel('Phase (º)')

    #ez
    ax_z1.set_title('$|E_{f_z}|^2$')
    pos_z1=ax_z1.imshow(np.abs(ez_XY)**2/Amp_max,extent=extent, interpolation='none', aspect='auto')
    ax_z1.set_xlabel('x (nm)')
    ax_z1.set_ylabel('y (nm)')
    ax_z1.axis('square')
    cbar_1_1=fig2.colorbar(pos_z1, ax=ax_z1)
    cbar_1_1.ax.set_ylabel('Relative intensity')
    
    ax_z2.set_title('$E_{f_z}$ phase')
    pos_z2=ax_z2.imshow(np.angle(ez_XY, deg=True)+180,extent=extent, interpolation='none', aspect='auto')
    ax_z2.set_xlabel('x (nm)')
    ax_z2.set_ylabel('y (nm)')
    ax_z2.axis('square')
    cbar_1_1=fig2.colorbar(pos_z2, ax=ax_z2,ticks=[5, 90,180,270,355])
    cbar_1_1.ax.set_ylabel('Phase (º)')
    fig2.tight_layout()

def plot_XY(ex_XY,ey_XY,ez_XY,x_range,figure_name):   
    '''
    Plot the calculated fields ont the XY plane
    '''
    plt.rcParams['font.size']=14
    #intensity plot
    fig, (ax2, ax3) = plt.subplots(num=str(figure_name)+': Intensity',figsize=(12, 4), ncols=2)

    x2,y2=np.shape(ex_XY)
    Ifield_xy=np.zeros((x2,y2))
    for i in range(x2):
        for j in range(y2):
            Ifield_xy[i,j]=np.real(ex_XY[i,j]*np.conj(ex_XY[i,j]) + ey_XY[i,j]*np.conj(ey_XY[i,j]) +ez_XY[i,j]*np.conj(ez_XY[i,j]))

    xmax=x_range/2
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
    Amp_max=np.abs(np.max([np.max(np.abs(ex_XY)),np.max(np.abs(ey_XY)),np.max(np.abs(ez_XY))]))
    #ex
    fig2, ((ax_x1,ax_y1,ax_z1),(ax_x2,ax_y2,ax_z2)) = plt.subplots(num=str(figure_name)+': Amplitudes',figsize=(18, 8),nrows=2, ncols=3)
    ax_x1.set_title('Ex amplitude')
    pos_x1=ax_x1.imshow(np.abs(ex_XY)/Amp_max,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_x1.set_xlabel('x (nm)')
    ax_x1.set_ylabel('y (nm)')
    ax_x1.axis('square')
    cbar_1_1=fig2.colorbar(pos_x1, ax=ax_x1)
    cbar_1_1.ax.set_ylabel('Relative amplitude')
    
    ax_x2.set_title('Ex phase')
    pos_x2=ax_x2.imshow(np.angle(ex_XY, deg=True)+180,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_x2.set_xlabel('x (nm)')
    ax_x2.set_ylabel('y (nm)')
    ax_x2.axis('square')
    cbar_1_1=fig2.colorbar(pos_x2, ax=ax_x2,ticks=[5, 90,180,270,355])
    cbar_1_1.ax.set_ylabel('Angle (Degrees)')

    #ey
    ax_y1.set_title('Ey amplitude')
    pos_y1=ax_y1.imshow(np.abs(ey_XY)/Amp_max,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_y1.set_xlabel('x (nm)')
    ax_y1.set_ylabel('y (nm)')
    ax_y1.axis('square')
    cbar_1_1=fig2.colorbar(pos_y1, ax=ax_y1)
    cbar_1_1.ax.set_ylabel('Relative amplitude')
    
    ax_y2.set_title('Ey phase')
    pos_y2=ax_y2.imshow(np.angle(ey_XY, deg=True)+180,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_y2.set_xlabel('x (nm)')
    ax_y2.set_ylabel('y (nm)')
    ax_y2.axis('square')
    cbar_1_1=fig2.colorbar(pos_y2, ax=ax_y2,ticks=[5, 90,180,270,355])
    cbar_1_1.ax.set_ylabel('Angle (Degrees)')

    #ez
    ax_z1.set_title('Ez amplitude')
    pos_z1=ax_z1.imshow(np.abs(ez_XY)/Amp_max,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_z1.set_xlabel('x (nm)')
    ax_z1.set_ylabel('y (nm)')
    ax_z1.axis('square')
    cbar_1_1=fig2.colorbar(pos_z1, ax=ax_z1)
    cbar_1_1.ax.set_ylabel('Relative amplitude')
    
    ax_z2.set_title('Ez phase')
    pos_z2=ax_z2.imshow(np.angle(ez_XY, deg=True)+180,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_z2.set_xlabel('x (nm)')
    ax_z2.set_ylabel('y (nm)')
    ax_z2.axis('square')
    cbar_1_1=fig2.colorbar(pos_z2, ax=ax_z2,ticks=[5, 90,180,270,355])
    cbar_1_1.ax.set_ylabel('Angle (Degrees)')
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.88)


