from matplotlib import pyplot as plt
import numpy as np

def plot_objective_field(Ifield, Ex, Ey, xmax, figure_name='', font_size=20, folder=''):
    #intensity and fit plot
    plt.rcParams['font.size']=20#tamaño de fuente
    fig1, (ax1, ax2) = plt.subplots(num=str(figure_name)+': Incident intensity',figsize=(12, 5), ncols=2)
    fig1.suptitle('Field at objective')

    ax1.set_title('Intensity')
    pos=ax1.imshow(Ifield,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
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
    #amplitude and phase plot 
    #ex
    fig2, ((ax_x1,ax_y1),(ax_x2,ax_y2)) = plt.subplots(num=str(figure_name)+': Incident amplitude',figsize=(12, 8),nrows=2, ncols=2)
    fig2.suptitle('Field at objective')
    ax_x1.set_title('ex amplitude')
    pos_x1=ax_x1.imshow(np.abs(Ex),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_x1.set_xlabel('x (mm)')
    ax_x1.set_ylabel('y (mm)')
    ax_x1.axis('square')
    cbar_1_1=fig2.colorbar(pos_x1, ax=ax_x1)
    cbar_1_1.ax.set_ylabel('Relative amplitude')
    
    ax_x2.set_title('ex phase')
    pos_x2=ax_x2.imshow(np.angle(Ex),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_x2.set_xlabel('x (mm)')
    ax_x2.set_ylabel('y (mm)')
    ax_x2.axis('square')
    cbar_1_1=fig2.colorbar(pos_x2, ax=ax_x2)
    cbar_1_1.ax.set_ylabel('Angle (Radians)')
    
    #ey
    ax_y1.set_title('ey amplitude')
    pos_y1=ax_y1.imshow(np.abs(Ey),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_y1.set_xlabel('x (mm)')
    ax_y1.set_ylabel('y (mm)')
    ax_y1.axis('square')
    cbar_1_1=fig2.colorbar(pos_y1, ax=ax_y1)
    cbar_1_1.ax.set_ylabel('Relative amplitude')
    
    ax_y2.set_title('ey phase')
    ax_y2.axis('square')
    pos_y2=ax_y2.imshow(np.angle(Ey),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_y2.set_xlabel('x (mm)')
    ax_y2.set_ylabel('y (mm)')
    ax_y2.axis('square')
    cbar_1_1=fig2.colorbar(pos_y2, ax=ax_y2)
    cbar_1_1.ax.set_ylabel('Angle (Radians)')
    
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.88)
    '''
    if figures are to be saved automatically:
    fig1.savefig(folder+'Intensity h = '+str(h)+', radius = '+str(radius)+' pre focus.png')
    fig2.savefig(folder+'Amplitude h = '+str(h)+', radius = '+str(radius)+' pre focus.png')
    '''    

def plot_in_cartesian(Ex,Ey,r_range,alpha,f,figure_name):
    '''
    Plot the fields Ex and Ey, who are described in the same coordinates as ex_lens and ey_lens. To do so the field in the closest cartesian coordinates for each position is calculated
    
    Args:
        :r_range: Radial distance in which to plot the field (total distance in x and y is 2*r_range)
    
        :alpha: Semi-angle of aperture of the objective lens, given by alpha=np.arcsin(NA / n)
        
        :f: Focal distance of the objective lens, given by the sine's law: f=h*n/NA
        
        :figure_name: Name for the ploted figures
    
    Returns:
        I_cartesian, Ex_cartesian, Ey_cartesian: Intensity and amplitude of the incident field calculated in cartesian coordinates
    '''
    divisions_phi,divisions_theta=np.shape(Ex)
    x_values=np.linspace(-r_range,r_range,2*divisions_theta) #positions along X in which to plot
    y_values=np.linspace(r_range,-r_range,2*divisions_theta)
    I_cartesian=np.zeros((2*divisions_theta,2*divisions_theta))
    Ex_cartesian=np.zeros((2*divisions_theta,2*divisions_theta),dtype=complex)
    Ey_cartesian=np.zeros((2*divisions_theta,2*divisions_theta),dtype=complex)
    
    #original polar coordinates in which the Ex and Ey where calculated:  
    theta_values=np.linspace(0,alpha,divisions_theta)  #divisions of theta in which the trapezoidal 2D integration is done
    rhop_values=np.sin(theta_values)*f              #given by the sine's law
    phip_values=np.linspace(0,2*np.pi,divisions_phi)   #divisions of phi in which the trapezoidal 2D integration is done
    
    #passage from polar to cartesian coordinates, keep in mind the positions are not to be exact since the plot is on a grid
    for i,x in enumerate(x_values):
        for j,y in enumerate(y_values):
            rhop=(x**2+y**2)**0.5
            phip=np.arctan2(y,x)
            if phip<0:
                phip+=2*np.pi
            if rhop<r_range:
                id_rho = (np.abs(rhop_values - rhop)).argmin() #get the closest indent for the coordinate in which the field was calculated
                id_phi = (np.abs(phip_values - phip)).argmin()
                Ex_cartesian[j,i]=Ex[id_phi,id_rho]
                Ey_cartesian[j,i]=Ey[id_phi,id_rho]

    I_cartesian=np.abs(Ex_cartesian)**2+np.abs(Ey_cartesian)**2
    
    #colorbar plot for the field:
    plt.rcParams['font.size']=10#tamaño de fuente
    fig1, (ax1, ax2) = plt.subplots(num=str(figure_name)+' Incident intensity',figsize=(9, 4), ncols=2)
    fig1.suptitle('Incident field intensity')
    
    # ax1.set_title('Intensity')
    pos=ax1.imshow(I_cartesian,extent=[-r_range,r_range,-r_range,r_range], interpolation='none', aspect='auto')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.axis('square')
    cbar1= fig1.colorbar(pos, ax=ax1)
    cbar1.ax.set_ylabel('Intensity (mW/cm\u00b2)')            
    #plot along the Y=0 axis:
    # ax2.set_title(' Intensity along x')
    ax2.plot(np.linspace(-r_range,r_range,2*divisions_theta),I_cartesian[divisions_theta,:])
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('Intensity  (mW/cm\u00b2)')  
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.90) 
    
    #Amplitude and phase plot 
    #Ex
    fig2, ((ax_x1,ax_y1),(ax_x2,ax_y2)) = plt.subplots(num=str(figure_name)+' Incident amplitude',figsize=(10, 8),nrows=2, ncols=2)
    fig2.suptitle('Incident field amplitude')
    ax_x1.set_title('$|E_{i_x}|^2$')
    pos_x1=ax_x1.imshow(np.abs(Ex_cartesian)**2,extent=[-r_range,r_range,-r_range,r_range], interpolation='none', aspect='equal')
    ax_x1.set_xlabel('x (mm)')
    ax_x1.set_ylabel('y (mm)')
    ax_x1.axis('square')
    cbar_1_1=fig2.colorbar(pos_x1, ax=ax_x1)
    cbar_1_1.ax.set_ylabel('Relative intensity')
    
    ax_x2.set_title('$E_{i_x}$ phase')
    pos_x2=ax_x2.imshow(np.angle(Ex_cartesian, deg=True),extent=[-r_range,r_range,-r_range,r_range], interpolation='none', aspect='equal')
    ax_x2.set_xlabel('x (mm)')
    ax_x2.set_ylabel('y (mm)')
    ax_x2.axis('square')
    cbar_1_1=fig2.colorbar(pos_x2, ax=ax_x2)
    cbar_1_1.ax.set_ylabel('Phase (degrees)')
    
    #Ey
    ax_y1.set_title('$|E_{i_y}|^2$')
    pos_y1=ax_y1.imshow(np.abs(Ey_cartesian)**2,extent=[-r_range,r_range,-r_range,r_range], interpolation='none', aspect='equal')
    ax_y1.set_xlabel('x (mm)')
    ax_y1.set_ylabel('y (mm)')
    ax_y1.axis('square')
    cbar_1_1=fig2.colorbar(pos_y1, ax=ax_y1)
    cbar_1_1.ax.set_ylabel('Relative intensity')
    
    ax_y2.set_title('$E_{i_y}$ phase')
    pos_y2=ax_y2.imshow(np.angle(Ey_cartesian, deg=True),extent=[-r_range,r_range,-r_range,r_range], interpolation='none', aspect='equal')
    ax_y2.set_xlabel('x (mm)')
    ax_y2.set_ylabel('y (mm)')
    ax_y2.axis('square')
    cbar_1_1=fig2.colorbar(pos_y2, ax=ax_y2)
    cbar_1_1.ax.set_ylabel('Phase (degrees)')
    
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.90)                
    
    return I_cartesian,Ex_cartesian,Ey_cartesian

