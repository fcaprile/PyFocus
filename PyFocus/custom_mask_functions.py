"""
Functions for the simulation of the foci obtained by an arbitrary phase mask
"""

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import time

def generate_incident_field(maskfunction,alpha,f,divisions_phi,divisions_theta,gamma,beta,w0,I0,wavelength):
    '''
    Generate a matrix for the field X and Y direction of the incident field on the lens, given the respective maskfunction
    
    Args:        
        :maskfunction (function): Analytical function that defines the phase mask, must be a function of the 5 internal variables: rho, phi, w0, f and k, with:
            
            rho: Radial coordinate from 0 to the aperture radius of the objective.
            
            phi: Azimutal coordinate from 0 to 2pi.
            
            w0: Radius of the incident gaussian beam.
            
            f: Focal distane of the objective lens (mm)
            
            k: Wavenumber in the objective lens medium (mm)
            
            The real part defines the amplitude of the incident field
    
        :divisions_phi,divisions_theta: Number of divisions in the phi and theta coordinates to use the 2D integration for the calculation of the focused field
 
        The rest of the parameters are specified in sim

    Returns:
        :arrays: ex_lens,ey_lens
        
    This arrays have the value of the amplitude of the incident field for each value of theta and phi: ex_lens[phi_position,theta_position]
    for phi_position a value in np.linspace(0,2*np.pi,divisions_phi) and theta_position a value in np.linspace(0,alpha,divisions_theta) 
    '''
    wavelength/=10**6#pasage from nm to mm
    k=2*np.pi/wavelength #k is given in nm, the same as wavelength
    ex_lens=np.zeros((divisions_phi,divisions_theta),dtype=complex)
    ey_lens=np.zeros((divisions_phi,divisions_theta),dtype=complex)

    theta_values=np.linspace(0,alpha,divisions_theta)  #divisions of theta in which the trapezoidal 2D integration is done
    rho_values=np.sin(theta_values)*f              #given by the sine's law
    phi_values=np.linspace(0,2*np.pi,divisions_phi)   #divisions of phi in which the trapezoidal 2D integration is done
    for i,phi in enumerate(phi_values):
        for j,rho in enumerate(rho_values):
            phase=maskfunction(rho,phi,w0,f,k)
            ex_lens[i,j]=phase
            ey_lens[i,j]=phase
    ex_lens*=np.cos(gamma*np.pi/180)*I0**0.5
    ey_lens*=np.sin(gamma*np.pi/180)*np.exp(1j*beta*np.pi/180)*I0**0.5
    
    return ex_lens,ey_lens

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

def custom_mask_objective_field(h,gamma,beta,divisions_theta,divisions_phi,N_rho,N_phi,alpha,focus,custom_field_function,R,L,I0,wavelength,w0,fig_name,plot=True):
    '''
    Calculate the incident field on the objective by fraunhofer's difraction formula for a custom phase mask

    The resultant matrix Ex and Ey are returned in polar coordinates (each row is a different value of phi and each column a different rho)  
    
    Args:    
        :N_rho and N_phi: Number of divisions for the calclation of the 2D integral in rho and phi respectively (this are not the coordinates in which the field is calculated)
        
        :divisions_theta,divisions_phi: Number of divisions for the field indicent on the lens on the theta and phi coordinates respectively. This field is later used to calculate the focused field.
        
        The rest of the parameters are specified in sim
    
    Returns:        
        :arrays: Ex, Ey: Amplitude of the incident field calculated on the pair of coordinates: theta_values=np.linspace(0,alpha,divisions_theta), phip_values=np.linspace(0,2*np.pi,divisions_phi)
    
        by using sine's law this corresponds to the radial positions rhop_values=np.sin(theta_values)*focus        
    
        :arrays: I_cartesian, Ex_cartesian, Ey_cartesian: Intensity and amplitude of the incident field calculated in cartesian coordinates
    '''
    wavelength/=10**6#passage of wavelength to mm
    
    print('Calculating incident field:')
    time.sleep(0.2)
    
    #define divisions for the integration:
    rho_values=np.linspace(0,R,N_rho)
    phi_values=np.linspace(0,2*np.pi,N_phi)
    rho,phi=np.meshgrid(rho_values,phi_values)
    
    #2D trapezoidal method weight:
    h_rho=R/N_rho
    h_phi=2*np.pi/N_phi
    weight_rho=np.zeros(N_rho)+h_rho
    weight_rho[0]=h_rho/2
    weight_rho[-1]=h_rho/2
    weight_phi=np.zeros(N_phi)+h_phi
    weight_phi[0]=h_phi/2
    weight_phi[-1]=h_phi/2
    weight=weight_rho*np.vstack(weight_phi)

    #define coordinates in which to calculate the field:    
    theta_values=np.linspace(0,alpha,divisions_theta)  #divisions of theta in which the trapezoidal 2D integration is done
    rhop_values=np.sin(theta_values)*focus              #given by the sine's law
    phip_values=np.linspace(0,2*np.pi,divisions_phi)   #divisions of phi in which the trapezoidal 2D integration is done

    #Define function to integrate and field matrix:    
    Ex=np.zeros((divisions_phi,divisions_theta),dtype=complex)
    kl=np.pi/wavelength/L
    '''
    #the function to integrate is:
    f=wavelength rho,phi: rho*custom_field_function(rho,phi)*np.exp(1j*(kl*(rho**2-2*rho*rhop*np.cos(phi-phip))))
    '''
    
    k=2*np.pi/wavelength
    #in order to save computing time, i do separatedly the calculation of terms that would otherwise e claculated multiple times, since they do not depend on rhop,phip (the coordinates at which the field is calculated)
    prefactor=rho*np.exp(1j*(k*L+kl*rho**2))*custom_field_function(rho,phi,w0,focus,k)*weight
    #numerical 2D integration: 
    for j in tqdm(range(divisions_phi)):
        phip=phip_values[j]
        for i,rhop in enumerate(rhop_values):
            phase=np.exp(1j*k*rhop**2/2/L)*np.exp(-1j*kl*(2*rho*rhop*np.cos(phi-phip)))         
            Ex[j,i]=np.sum(prefactor*phase)
    
    Ex*=-1j/wavelength/L
    
    #on this approximation, the field in the Y direction is the same as the field on the X direction with a different global phase and amplitude        
    Ey=Ex*np.exp(1j*np.pi/180*beta)
    Ex*=np.cos(np.pi/180*gamma)*I0**0.5
    Ey*=np.sin(np.pi/180*gamma)*I0**0.5
    
    I_cartesian,Ex_cartesian,Ey_cartesian=plot_in_cartesian(Ex,Ey,h,alpha,focus,fig_name)
        
    return Ex,Ey,I_cartesian,Ex_cartesian,Ey_cartesian


def custom_mask_focus_field_XY(ex_lens,ey_lens,alpha,h,wavelength,zp0,resolution_x,divisions_theta,divisions_phi,x_range,countdown=True,x0=0,y0=0):
    '''
    2D integration to calculate the field focused by a high aperture lens on the XY plane
    
    Args:        
        :ex_lens,ey_lens: X and Y component of the incident field. This arrays have the value of the amplitude of the incident field for each value of theta and phi: ex_lens[phi_position,theta_position] for phi_position a value in np.linspace(0,2*np.pi,divisions_phi) and theta_position a value in np.linspace(0,alpha,divisions_theta) 

        :zp0: Axial position for the XY plane (given by z=zp0)
    
        :resolution_x: Resolution for the field at the focus, the same for x and y
        
        :divisions_theta,divisions_phi: Resolution for the 2D calculus (must be the same as the size of ex_lens and ey_lens) 
    
        :wavelength: Wavelength (nm) in the medium (equals wavelength in vacuum/n) 
        
        :x0, y0: Used for centering the XY field at an x0, y0 position (nm)

        :countdown (bool): True for shoing a progress bar with the time elapsed and expected to finish the calculation, True not recomended if this function is to be used many times

        The rest of the parameters are specified in sim
    
    Returns:
        :arrays: ex,ey,ez: Cartesian components of the focused field on the XY plane, given by z=zp0
    '''
    
    if countdown==True:
        print('Calculating field near the focus:')
        time.sleep(0.5)
    
    #passage from mm to nm and sine's law  
    focus=h/np.sin(alpha)*10**6
        
    #defining functions for rotating in the azimutal direction pi and 3*pi/2, which corresponds to changing the order of the columns in ex_lens and ey_lens since the columns are given by an azimutal position
    def rotate_180º(matrix):
        a,b=np.shape(matrix)       
        aux=np.zeros((a,b),dtype=complex)        
        for i in range(a):
            aux[i-int(a/2),:]=matrix[i,:]
        return aux
    
    #The Y component of incident field must be evaluated at phi-pi/2, which is equivalent to moving the rows of the matrix    
    def rotate_270º(matrix):
        a,b=np.shape(matrix)       
        aux=np.zeros((a,b),dtype=complex)        
        for i in range(a):
            aux[i-int(3*a/4),:]=matrix[i,:]
        return aux
    
    #To use the incident field as the function to be integrated in the equations deried from born and wolf, the incident field must be evaluated at phi-pi for the X component and at phi-3*pi/2 for the Y component. 
    #The motives for this correspond to a difference in the coordinate system used when deriving the equations, where the z versor points in the oposite direction
    #This fact is not described in the paper since i am not 100% sure this is the motive, but the rotation is necesary to obtain the needed result in all my tests with custom masks
    ex_lens=rotate_180º(ex_lens)
    ey_lens=rotate_270º(ey_lens)
    
    '''
    # the functions i am going to integrate are:
    fun1=wavelength phi,theta: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun2=wavelength phi,theta: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun3=wavelength phi,theta: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun4=wavelength phi,theta: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun5=wavelength phi,theta: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun6=wavelength phi,theta: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    '''  
    #2D trapezoidal method weight:
    h_theta=alpha/divisions_theta
    h_phi=2*np.pi/divisions_phi
    weight_trapezoid_rho=np.zeros(divisions_theta)+h_theta
    weight_trapezoid_rho[0]=h_theta/2
    weight_trapezoid_rho[-1]=h_theta/2
    weight_trapezoid_phi=np.zeros(divisions_phi)+h_phi
    weight_trapezoid_phi[0]=h_phi/2
    weight_trapezoid_phi[-1]=h_phi/2
    weight_trapezoid=weight_trapezoid_rho*np.vstack(weight_trapezoid_phi)#represents the area of each trapezoid for each position in phi,theta
    
    #define coordinates in which to calculate the field:    
    xmax=x_range/2
    x_values=np.linspace(-xmax+x0,xmax+x0,resolution_x)
    y_values=np.linspace(xmax+y0,-xmax+y0,resolution_x)
    ex=np.zeros((resolution_x,resolution_x),dtype=complex)
    ey=np.copy(ex)
    ez=np.copy(ex)
    
    #define divisions for the integration:
    theta_values=np.linspace(0,alpha,divisions_theta)    #divisions of theta in which the trapezoidal 2D integration is done
    phi_values=np.linspace(0,2*np.pi,divisions_phi)      #divisions of phi in which the trapezoidal 2D integration is done
    theta,phi=np.meshgrid(theta_values,phi_values)        #turn the divisions into a matrix in order to apply the function more easily

    kz=zp0*2*np.pi/wavelength

    #now begins the integration, in order to save computing time i do the trigonometric functions separatedly and save the value into another variable
    cos_theta=np.cos(theta)
    cos_theta_sqrt=cos_theta**0.5
    sin_theta=np.sin(theta)
    cos_phi=np.cos(phi)
    sin_phi=np.sin(phi)
    sin_phi_square=sin_phi**2
    
    prefactor_general=cos_theta_sqrt*sin_theta
    prefactor_x=prefactor_general*(cos_theta+(1-cos_theta)*sin_phi_square)
    prefactor_y=prefactor_general*(1-cos_theta)*cos_phi*sin_phi
    prefactor_z=prefactor_general*sin_theta*cos_phi
    
    Axx=prefactor_x*ex_lens*weight_trapezoid
    Axy=prefactor_y*ex_lens*weight_trapezoid
    Axz=prefactor_z*ex_lens*weight_trapezoid

    Ayx=prefactor_y*ey_lens*weight_trapezoid
    Ayy=-prefactor_x*ey_lens*weight_trapezoid
    Ayz=prefactor_z*ey_lens*weight_trapezoid

    cos_theta_kz=cos_theta*kz
    #now for each position in which i calculate the field i do the integration
    if countdown==True:
        for i in tqdm(range(resolution_x),desc='XY plane'):
            x=x_values[i]
            for j,y in enumerate(y_values):#idea, rotar en phi es correr las columnas de la matriz ex, ey
                rhop=(x**2+y**2)**0.5
                phip=np.arctan2(y,x)
                kr=rhop*2*np.pi/wavelength
                sin_theta_kr=sin_theta*kr
                phase_inc_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip) + cos_theta_kz))#phase for the X incident component
                phase_inc_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip) + cos_theta_kz))#phase for the Y incident component
                #now, the integration is made as the sum of the value of the integrand in each position of phi,theta:
                ex[j,i]=np.sum(Axx*phase_inc_x)+np.sum(Ayx*phase_inc_y) 
                ey[j,i]=np.sum(Axy*phase_inc_x)+np.sum(Ayy*phase_inc_y)
                ez[j,i]=np.sum(Axz*phase_inc_x)+np.sum(Ayz*phase_inc_y)
    else:
        for i,x in enumerate(x_values):
            for j,y in enumerate(y_values):
                rhop=(x**2+y**2)**0.5
                phip=np.arctan2(y,x)
                kr=rhop*2*np.pi/wavelength
                sin_theta_kr=sin_theta*kr
                phase_inc_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip) + cos_theta_kz))#phase for the X incident component
                phase_inc_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip) + cos_theta_kz))#phase for the Y incident component
                #now, the integration is made as the sum of the value of the integrand in each position of phi,theta:
                ex[j,i]=np.sum(Axx*phase_inc_x)+np.sum(Ayx*phase_inc_y)
                ey[j,i]=np.sum(Axy*phase_inc_x)+np.sum(Ayy*phase_inc_y)
                ez[j,i]=np.sum(Axz*phase_inc_x)+np.sum(Ayz*phase_inc_y)
        
    ex*=-1j*focus/wavelength
    ey*=1j*focus/wavelength
    ez*=1j*focus/wavelength
    
    return ex,ey,ez

def custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,wavelength,z_range,resolution_z,zp0,resolution_x,divisions_theta,divisions_phi,x_range,x0=0,y0=0,plot_plane='X'):
    '''
    2D integration to calculate the field focused by a high aperture lens on the XY and XZ plane
    
    Args:        
        :ex_lens,ey_lens: X and Y component of the incident field. This arrays have the value of the amplitude of the incident field for each value of theta and phi: ex_lens[phi_position,theta_position] for phi_position a value in np.linspace(0,2*np.pi,divisions_phi) and theta_position a value in np.linspace(0,alpha,divisions_theta) 
        
        :zp0: Axial position for the XY plane (given by z=zp0)
    
        :resolution_x: Resolution for the field at the focus, in the x and y coordinates
        
        :resolution_z: Number of pixels for the field at the focus, in the z coordinate
        
        :divisions_theta,divisions_phi: Resolution for the 2D calculus (must be the same as the size of ex_lens and ey_lens) 
    
        :wavelength: Wavelength (nm) in the medium (equals wavelength in vacuum/n) 
        
        :x0, y0: Used for centering the XY field at an x0, y0 position (nm)
        
        :plot_plane (string): Available values: 'X' or 'Y', select to plot the ZX or the ZY plane respectivelly

        The rest of the parameters are specified in sim
    
    Returns:        
        :arrays: Ex_XZ,Ey_XZ,Ez_XZ,Ex_XY,Ey_XY,Ez_XY, each one is a matrix with the amplitude of each cartesian component on the XZ plane (Ex,Ey,Ez) or on the XY plane (Ex2,Ey2,Ez2)
    
        Each index of the matrixes corresponds to a different pair of coordinates, for example: 
                        
        Ex_XZ[z,x] with z each index of the coordinates np.linspace(z_range/2,-z_range/2,resolution_z) and x each index for np.linspace(-x_range/2**0.5,x_range/2**0.5,resolution_x) in which the field is calculated, the x range is sqrt(2) times longer for consistency with the VP and no_mask functions
        
        Ex_XY[y,x2] with y each index of the coordinates np.linspace(x_range/2,-x_range/2,resolution_x) and x each index for np.linspace(-x_range/2,x_range/2,resolution_x) in which the field is calculated
        
        The XZ plane is given by y=0 and the XZ plane by z=zp0 
    '''
    #XY plane: 
    Ex_XY,Ey_XY,Ez_XY=custom_mask_focus_field_XY(ex_lens,ey_lens,alpha,h,wavelength,zp0,resolution_x,divisions_theta,divisions_phi,x_range,countdown=True,x0=x0,y0=y0)
    
    #XZ plane:
    if int(resolution_z%2)==0:
        resolution_z+=1    #make the middle coordinate on Z be Z=0
        
    #passage from mm to nm and sine's law  
    focus=h/np.sin(alpha)*10**6
        
    def rotate_180º(matrix):
        a,b=np.shape(matrix)       
        aux=np.zeros((a,b),dtype=complex)        
        for i in range(a):
            aux[i-int(a/2),:]=matrix[i,:]
        return aux
    
    #The Y component of incident field must be evaluated at phi-pi/2, which is equivalent to moving the rows of the matrix    
    def rotate_270º(matrix):
        a,b=np.shape(matrix)       
        aux=np.zeros((a,b),dtype=complex)        
        for i in range(a):
            aux[i-int(3*a/4),:]=matrix[i,:]
        return aux
    
    #To use the incident field as the function to be integrated in the equations deried from born and wolf, the incident field must be evaluated at phi-pi for the X component and at phi-3*pi/2 for the Y component. 
    #The motives for this correspond to a difference in the coordinate system used when deriving the equations, where the z versor points in the oposite direction
    #This fact is not described in the paper since i am not 100% sure this is the motive, but the rotation is necesary to obtain the needed result in all my tests with custom masks
    ex_lens=rotate_180º(ex_lens)
    ey_lens=rotate_270º(ey_lens)

    '''
    # the functions i am going to integrate are:
    fun1=wavelength theta, phi: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun2=wavelength theta, phi: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun3=wavelength theta, phi: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun4=wavelength theta, phi: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun5=wavelength theta, phi: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun6=wavelength theta, phi: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    '''  
        
    #2D trapezoidal method weight:
    h_theta=alpha/divisions_theta
    h_phi=2*np.pi/divisions_phi
    weight_trapezoid_rho=np.zeros(divisions_theta)+h_theta
    weight_trapezoid_rho[0]=h_theta/2
    weight_trapezoid_rho[-1]=h_theta/2
    weight_trapezoid_phi=np.zeros(divisions_phi)+h_phi
    weight_trapezoid_phi[0]=h_phi/2
    weight_trapezoid_phi[-1]=h_phi/2
    weight_trapezoid=weight_trapezoid_rho*np.vstack(weight_trapezoid_phi) #represents the area of each trapezoid for each position in phi,theta

    #define coordinates in which to calculate the field:                
    xmax=x_range
    x_values=np.linspace(-xmax+x0,xmax+x0,resolution_x)/(2**0.5) #on the XZ plane the x axis is calculated sqrt(2) times longer in order to make it analogue to the VPP and no_mask functions, since the maximum radial distance is larger than the maximum X or Y distance in the XY plane
    z_values=np.linspace(z_range/2,-z_range/2,resolution_z)
    
    Ex_XZ=np.zeros((resolution_z,resolution_x),dtype=complex)
    Ey_XZ=np.copy(Ex_XZ)
    Ez_XZ=np.copy(Ex_XZ)
        
    #define divisions for the integration:
    theta_values=np.linspace(0,alpha,divisions_theta)    #divisions of theta in which the trapezoidal 2D integration is done
    phi_values=np.linspace(0,2*np.pi,divisions_phi)      #divisions of phi in which the trapezoidal 2D integration is done
    theta,phi=np.meshgrid(theta_values,phi_values)        #turn the divisions into a matrix in order to apply the function more easily
    
    #now begins the integration, in order to save computing time i do the trigonometric functions separatedly and save the value into another variable
    cos_theta=np.cos(theta)
    cos_theta_sqrt=cos_theta**0.5
    sin_theta=np.sin(theta)
    cos_phi=np.cos(phi)
    sin_phi=np.sin(phi)
    sin_phi_square=sin_phi**2
    
    prefactor_general=cos_theta_sqrt*sin_theta
    prefactor_x=prefactor_general*(cos_theta+(1-cos_theta)*sin_phi_square)
    prefactor_y=prefactor_general*(1-cos_theta)*cos_phi*sin_phi
    prefactor_z=prefactor_general*sin_theta*cos_phi

    Axx=prefactor_x*ex_lens*weight_trapezoid
    Axy=prefactor_y*ex_lens*weight_trapezoid
    Axz=prefactor_z*ex_lens*weight_trapezoid

    Ayx=prefactor_y*ey_lens*weight_trapezoid
    Ayy=-prefactor_x*ey_lens*weight_trapezoid
    Ayz=prefactor_z*ey_lens*weight_trapezoid

    if plot_plane=='X':
        for j in tqdm(range(resolution_z),desc='XZ plane'):
            zp0=z_values[j]
            for i,x in enumerate(x_values):#idea, rotar en phi es correr las columnas de la matriz ex, ey
                rhop=np.abs(x)
                phip=np.arctan2(0,x)
                kr=rhop*2*np.pi/wavelength
                kz=zp0*2*np.pi/wavelength
                sin_theta_kr=sin_theta*kr
                cos_theta_kz=cos_theta*kz
                phase_inc_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip) + cos_theta_kz))#phase for the X incident component
                phase_inc_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip) + cos_theta_kz))#phase for the Y incident component
                #now, the integration is made as the sum of the value of the integrand in each position of phi,theta:
                Ex_XZ[j,i]=np.sum(Axx*phase_inc_x)+np.sum(Ayx*phase_inc_y)
                Ey_XZ[j,i]=np.sum(Axy*phase_inc_x)+np.sum(Ayy*phase_inc_y)
                Ez_XZ[j,i]=np.sum(Axz*phase_inc_x)+np.sum(Ayz*phase_inc_y)
    else:
        if plot_plane=='Y':           
            for j in tqdm(range(resolution_z),desc='YZ plane'):
                zp0=z_values[j]
                for i,y in enumerate(x_values):#idea, rotar en phi es correr las columnas de la matriz ex, ey
                    rhop=np.abs(y)
                    phip=np.arctan2(y,0)
                    kr=rhop*2*np.pi/wavelength
                    kz=zp0*2*np.pi/wavelength
                    sin_theta_kr=sin_theta*kr
                    cos_theta_kz=cos_theta*kz
                    phase_inc_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip) + cos_theta_kz))#phase for the X incident component
                    phase_inc_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip) + cos_theta_kz))#phase for the Y incident component
                    #now, the integration is made as the sum of the value of the integrand in each position of phi,theta:
                    Ex_XZ[j,i]=np.sum(Axx*phase_inc_x)+np.sum(Ayx*phase_inc_y)
                    Ey_XZ[j,i]=np.sum(Axy*phase_inc_x)+np.sum(Ayy*phase_inc_y)
                    Ez_XZ[j,i]=np.sum(Axz*phase_inc_x)+np.sum(Ayz*phase_inc_y)
        else:
            print('Options for plot_plane are \'X\' and \'Y\' ')
    Ex_XZ*=-1j*focus/wavelength
    Ey_XZ*=1j*focus/wavelength
    Ez_XZ*=1j*focus/wavelength
    
    return Ex_XZ,Ey_XZ,Ez_XZ,Ex_XY,Ey_XY,Ez_XY

