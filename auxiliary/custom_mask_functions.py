"""
Functions for the simulation of the foci obtained by an arbitrary phase mask
"""

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import time

def generate_incident_field(maskfunction,alpha,focus,resolution_phi,resolution_theta,gamma,beta,w0,I0,wavelength):
    '''
    Generate a matrix for the field X and Y direction of the incident field on the lens, given the respective maskfunction
    
    Parameters:
        
    maskfunction (function): analytical function that defines the phase mask. The real part also defines the incident field's amplitude
    
    resolution_phi,resolution_theta: number of divisions in the phi and theta coordinates to use the 2D integration for the calculation of the focused field
 
    The rest of the parameters are specified in sim

    This arrays have the value of the amplitude of the incident field for each value of theta and phi: ex_lens[phi_position,theta_position]
    for phi_position a value in np.linspace(0,2*np.pi,resolution_phi) and theta_position a value in np.linspace(0,alpha,resolution_theta) 
    '''

    k=2*np.pi/wavelength #k is given in nm, the same as wavelength
    ex_lens=np.zeros((resolution_phi,resolution_theta),dtype=complex)
    ey_lens=np.zeros((resolution_phi,resolution_theta),dtype=complex)

    theta_values=np.linspace(0,alpha,resolution_theta)  #divisions of theta in which the trapezoidal 2D integration is done
    rhop_values=np.sin(theta_values)*focus              #given by the sine's law
    phip_values=np.linspace(0,2*np.pi,resolution_phi)   #divisions of phi in which the trapezoidal 2D integration is done
    for i,phip in enumerate(phip_values):
        for j,rhop in enumerate(rhop_values):
            phase=maskfunction(rhop,phip,w0,focus,k)
            ex_lens[i,j]=phase
            ey_lens[i,j]=phase
    ex_lens*=np.cos(gamma*np.pi/180)*I0**0.5
    ey_lens*=np.sin(gamma*np.pi/180)*np.exp(1j*beta*np.pi/180)*I0**0.5
    
    return ex_lens,ey_lens

def plot_in_cartesian(Ex,Ey,xmax,alpha,focus,figure_name):
    '''
    Plot the fields Ex and Ey, who are described in polar coordinates. To do so the field in the closest cartesian coordinates for each position is calculated
    
    Returns:
        I_cartesian, Ex_cartesian, Ey_cartesian: Intensity and amplitude of the incident field calculated in cartesian coordinates
    '''
    resolution_phi,resolution_theta=np.shape(Ex)
    x_values=np.linspace(-xmax,xmax,2*resolution_theta) #positions along X in which to plot
    y_values=np.linspace(xmax,-xmax,2*resolution_theta)
    I_cartesian=np.zeros((2*resolution_theta,2*resolution_theta))
    Ex_cartesian=np.zeros((2*resolution_theta,2*resolution_theta),dtype=complex)
    Ey_cartesian=np.zeros((2*resolution_theta,2*resolution_theta),dtype=complex)
    
    #original polar coordinates in which the Ex and Ey where calculated:  
    theta_values=np.linspace(0,alpha,resolution_theta)  #divisions of theta in which the trapezoidal 2D integration is done
    rhop_values=np.sin(theta_values)*focus              #given by the sine's law
    phip_values=np.linspace(0,2*np.pi,resolution_phi)   #divisions of phi in which the trapezoidal 2D integration is done
    
    #passage from polar to cartesian coordinates, keep in mind the positions are not to be exact since the plot is on a grid
    for i,x in enumerate(x_values):
        for j,y in enumerate(y_values):
            rhop=(x**2+y**2)**0.5
            phip=np.arctan2(y,x)
            if phip<0:
                phip+=2*np.pi
            if rhop<xmax:
                id_rho = (np.abs(rhop_values - rhop)).argmin() #get the closest indent for the coordinate in which the field was calculated
                id_phi = (np.abs(phip_values - phip)).argmin()
                Ex_cartesian[j,i]=Ex[id_phi,id_rho]
                Ey_cartesian[j,i]=Ey[id_phi,id_rho]

    I_cartesian=np.abs(Ex_cartesian)**2+np.abs(Ey_cartesian)**2
    
    #colorbar plot for the field:
    plt.rcParams['font.size']=12#tamaño de fuente
    fig1, (ax1, ax2) = plt.subplots(num=str(figure_name)+': Incident intensity',figsize=(12, 5), ncols=2)
    fig1.suptitle('Field at the objective using 2D integration')
    
    ax1.set_title('Intensity')
    pos=ax1.imshow(I_cartesian,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.axis('square')
    cbar1= fig1.colorbar(pos, ax=ax1)
    cbar1.ax.set_ylabel('Intensity (kW/cm\u00b2)')            
    #plot along the Y=0 axis:
    ax2.set_title(' Intensity along x')
    ax2.plot(np.linspace(-xmax,xmax,2*resolution_theta),I_cartesian[resolution_theta,:])
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('Intensity  (kW/cm\u00b2)')  
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.80)                
    
    #Amplitude and phase plot 
    #Ex
    fig2, ((ax_x1,ax_y1),(ax_x2,ax_y2)) = plt.subplots(num=str(figure_name)+': Incident amplitude',figsize=(12, 8),nrows=2, ncols=2)
    ax_x1.set_title('ex amplitude')
    pos_x1=ax_x1.imshow(np.abs(Ex_cartesian),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='equal')
    ax_x1.set_xlabel('x (mm)')
    ax_x1.set_ylabel('y (mm)')
    ax_x1.axis('square')
    cbar_1_1=fig2.colorbar(pos_x1, ax=ax_x1)
    cbar_1_1.ax.set_ylabel('Amplitude')
    
    ax_x2.set_title('ex phase')
    pos_x2=ax_x2.imshow(np.angle(Ex_cartesian),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='equal')
    ax_x2.set_xlabel('x (mm)')
    ax_x2.set_ylabel('y (mm)')
    ax_x2.axis('square')
    cbar_1_1=fig2.colorbar(pos_x2, ax=ax_x2)
    cbar_1_1.ax.set_ylabel('Angle (Radians)')
    
    #Ey
    ax_y1.set_title('ey amplitude')
    pos_y1=ax_y1.imshow(np.abs(Ey_cartesian),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='equal')
    ax_y1.set_xlabel('x (mm)')
    ax_y1.set_ylabel('y (mm)')
    ax_y1.axis('square')
    cbar_1_1=fig2.colorbar(pos_y1, ax=ax_y1)
    cbar_1_1.ax.set_ylabel('Amplitude')
    
    ax_y2.set_title('ey phase')
    pos_y2=ax_y2.imshow(np.angle(Ey_cartesian),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='equal')
    ax_y2.set_xlabel('x (mm)')
    ax_y2.set_ylabel('y (mm)')
    ax_y2.axis('square')
    cbar_1_1=fig2.colorbar(pos_y2, ax=ax_y2)
    cbar_1_1.ax.set_ylabel('Angle (Radians)')
    
    fig2.tight_layout()
    
    return I_cartesian,Ex_cartesian,Ey_cartesian

def custom_mask_objective_field(h,gamma,beta,resolution_theta,resolution_phi,N_rho,N_phi,alpha,focus,mask_function,R,L,I0,wavelength,w0,fig_name,plot=True):
    '''
    Calculate the incident field on the objective by fraunhofer's difraction formula for a custom phase mask

    The resultant matrix Ex and Ey are returned in polar coordinates (each row is a different value of phi and each column a different rho)  
    
    Parameters:
    
        N_rho and N_phi: Number of divisions for the calclation of the 2D integral in rho and phi respectively (this are not the coordinates in which the field is calculated)
        
        resolution_theta,resolution_phi: Number of divisions for the field indicent on the lens on the theta and phi coordinates respectively. This field is later used to calculate the focused field.
        

    The rest of the parameters are specified in sim
    
    Returns:
        
        Ex, Ey (arrays): Amplitude of the incident field calculated on the pair of coordinates: theta_values=np.linspace(0,alpha,resolution_theta), phip_values=np.linspace(0,2*np.pi,resolution_phi)
    
    by using sine's law this corresponds to the radial positions rhop_values=np.sin(theta_values)*focus        
    
        I_cartesian, Ex_cartesian, Ey_cartesian: Intensity and amplitude of the incident field calculated in cartesian coordinates
    '''
    
    print('Calculating field at the objective:')
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
    theta_values=np.linspace(0,alpha,resolution_theta)  #divisions of theta in which the trapezoidal 2D integration is done
    rhop_values=np.sin(theta_values)*focus              #given by the sine's law
    phip_values=np.linspace(0,2*np.pi,resolution_phi)   #divisions of phi in which the trapezoidal 2D integration is done

    #Define function to integrate and field matrix:    
    Ex=np.zeros((resolution_phi,resolution_theta),dtype=complex)
    kl=np.pi/wavelength/L
    '''
    #the function to integrate is:
    f=wavelength rho,phi: rho*mask_function(rho,phi)*np.exp(1j*(kl*(rho**2-2*rho*rhop*np.cos(phi-phip))))
    '''
    
    k=2*np.pi/wavelength
    #in order to save computing time, i do separatedly the calculation of terms that would otherwise e claculated multiple times, since they do not depend on rhop,phip (the coordinates at which the field is calculated)
    prefactor=rho*np.exp(1j*(k*L+kl*rho**2))*mask_function(rho,phi,w0,focus,k)*weight
    #numerical 2D integration: 
    for j in tqdm(range(resolution_phi)):
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

def test_custom_mask_objective_field(psi,delta,resolution_theta,resolution_phi,N_rho,N_phi,alpha,mask_function,h,L,I0,wavelength,w0,plot=True):
    '''
    Quick run for testing if the resolution used to do the 2D integration is high enought.
    
    Meant for difraction for an arbitrary phase mask under the paraxial approximation, using the GPU
    '''
    print('Calculating field at the objective:')
    time.sleep(0.2)
    focus=h/np.sin(alpha)
    
    #define divisions for the integration:
    rho_values=np.linspace(0,h,N_rho)
    phi_values=np.linspace(0,2*np.pi,N_phi)
    rho,phi=np.meshgrid(rho_values,phi_values)
    
    #2D trapezoidal method weight:
    h_rho=h/N_rho
    h_phi=2*np.pi/N_phi
    weight_rho=np.zeros(N_rho)+h_rho
    weight_rho[0]=h_rho/2
    weight_rho[-1]=h_rho/2
    weight_phi=np.zeros(N_phi)+h_phi
    weight_phi[0]=h_phi/2
    weight_phi[-1]=h_phi/2
    weight=weight_rho*np.vstack(weight_phi)
    
    resolution_phi=int(resolution_phi/20) #only make 1/20 of the total calculation
    #define coordinates in which to calculate the field:    
    theta_values=np.linspace(0,alpha,resolution_theta)    
    rhop_values=np.sin(theta_values)*focus
    phip_values=np.linspace(0,np.pi/20,resolution_phi)

    #Define function to integrate and field matrix:    
    Ex=np.zeros((resolution_phi,resolution_theta),dtype=complex)
    kl=np.pi/wavelength/L
    '''
    #the function to integrate is:
    f=wavelength rho,phi: rho*mask_function(rho,phi)*np.exp(1j*(kl*(rho**2-2*rho*rhop*np.cos(phi-phip))))
    '''
    k=2*np.pi/wavelength
    #in order to save computing time, i do separatedly the calculation of terms that would otherwise e claculated multiple times, since they do not depend on rhop,phip (the coordinates at which the field is calculated)
    prefactor=rho*mask_function(rho,phi)*weight

    #numerical 2D integration: 
    for j in tqdm(range(resolution_phi)):
        phip=phip_values[j]
        for i,rhop in enumerate(rhop_values):
            phase=np.exp(1j*k*(L+rhop**2/2/L))*np.exp(1j*(kl*(rho**2-2*rho*rhop*np.cos(phi-phip))))         
            Ex[j,i]=np.sum(prefactor*phase)
    
    Ex*=-1j/wavelength/L

    #on this approximation, the field in the Y direction is the same as the field on the X direction with a different global phase and amplitude        
    Ey=Ex*np.exp(1j*np.pi/180*delta)
    Ex*=np.cos(np.pi/180*psi)*I0**0.5
    Ey*=np.sin(np.pi/180*psi)*I0**0.5

    I_cartesian,Ex_cartesian,Ey_cartesian=plot_in_cartesian(Ex,Ey,h,alpha,focus)

    return Ex,Ey,I_cartesian,Ex_cartesian,Ey_cartesian

def custom_mask_focus_field_XY(ex_lens,ey_lens,alpha,h,wavelength,zp0,resolution_focus,resolution_theta,resolution_phi,FOV_focus,countdown=True,x0=0,y0=0):
    '''
    2D integration to calculate the field focused by a high aperture lens on the XY plane
    
    Parameters:
        
        ex_lens,ey_lens: X and Y component of the incident field
        
        zp0: Axial position for the XY plane (given by z=zp0)
    
        resolution_focus: Resolution for the field at the focus, the same for x and y
        
        resolution_theta,resolution_phi: Resolution for the 2D calculus (must be the same as the size of ex_lens and ey_lens) 
    
        wavelength: Wavelength (nm) in the medium (equals wavelength in vacuum/n) 
        
        x0 and y0: Used for centering the XY field at an x0, y0 position

        countdown (bool): True means you are only running this fuction once and you want to see a progress bar with the time elapsed and expected to finish the calculation

    The rest of the parameters are specified in sim
    
    Returns:
        ex,ey,ez (arrays): Cartesian components of the focused field on the XY plane, given by z=zp0
    '''
    
    if countdown==True:
        print('Calculating field at the focal plane:')
        time.sleep(0.5)
    
    #passage from mm to nm and sine's law  
    focus=h/np.sin(alpha)*10**6
        
    #The Y component of incident field must be evaluated at phi-pi-pi/2, which is equivalent to moving the rows of the matrix    
    def rotate_90º(matrix):
        a,b=np.shape(matrix)       
        aux=np.zeros((a,b),dtype=complex)        
        for i in range(a):
            aux[i-int(3*a/4),:]=matrix[i,:]
        return aux

    ey_lens=rotate_90º(ey_lens)
    
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
    h_theta=alpha/resolution_theta
    h_phi=2*np.pi/resolution_phi
    weight_trapezoid_rho=np.zeros(resolution_theta)+h_theta
    weight_trapezoid_rho[0]=h_theta/2
    weight_trapezoid_rho[-1]=h_theta/2
    weight_trapezoid_phi=np.zeros(resolution_phi)+h_phi
    weight_trapezoid_phi[0]=h_phi/2
    weight_trapezoid_phi[-1]=h_phi/2
    weight_trapezoid=weight_trapezoid_rho*np.vstack(weight_trapezoid_phi)#represents the area of each trapezoid for each position in phi,theta
    
    #define coordinates in which to calculate the field:    
    xmax=FOV_focus/2
    x_values=np.linspace(-xmax+x0,xmax+x0,resolution_focus)
    y_values=np.linspace(xmax+y0,-xmax+y0,resolution_focus)
    ex=np.zeros((resolution_focus,resolution_focus),dtype=complex)
    ey=np.copy(ex)
    ez=np.copy(ex)
    
    #define divisions for the integration:
    theta_values=np.linspace(0,alpha,resolution_theta)    #divisions of theta in which the trapezoidal 2D integration is done
    phi_values=np.linspace(0,2*np.pi,resolution_phi)      #divisions of phi in which the trapezoidal 2D integration is done
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

    Ayx=-prefactor_y*ey_lens*weight_trapezoid
    Ayy=prefactor_x*ey_lens*weight_trapezoid
    Ayz=-prefactor_z*ey_lens*weight_trapezoid

    cos_theta_kz=cos_theta*kz
    #now for each position in which i calculate the field i do the integration
    if countdown==True:
        for i in tqdm(range(resolution_focus),desc='XY plane'):
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

def custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,wavelength,z_FOV,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,FOV_focus,x0=0,y0=0,plot_plane='X'):
    '''
    2D integration to calculate the field focused by a high aperture lens on the XY and XZ plane
    
        ex_lens,ey_lens: X and Y component of the incident field
        
        zp0: Axial position for the XY plane (given by z=zp0)
    
        resolution_focus: Resolution for the field at the focus, the same for x and y
        
        resolution_theta,resolution_phi: Resolution for the 2D calculus (must be the same as the size of ex_lens and ey_lens) 
    
        wavelength: Wavelength (nm) in the medium (equals wavelength in vacuum/n) 
        
        x0 and y0: Used for centering the XY field at an x0, y0 position

        countdown (bool): True means you are only running this fuction once and you want to see a progress bar with the time elapsed and expected to finish the calculation

    The rest of the parameters are specified in sim
    
    Returns:
        arrays: Ex_XZ,Ey_XZ,Ez_XZ,Ex_XY,Ey_XY,Ez_XY, each one is a matrix with the amplitude of each cartesian component on the XZ plane (ex_XZ,ey_XZ,ez_XZ) or on the XY plane (ex_XY,ey_XY,ez_XY)
    
    Each index of the matrixes corresponds to a different pair of coordinates, for example: 
        
    ex_XZ[z,x] with z each index of the coordinates np.linspace(z_field_of_view/2,-z_field_of_view/2,2*int(z_field_of_view/zsteps/2)) and x each index for np.linspace(-field_of_view/2**0.5,field_of_view/2**0.5,2*int(field_of_view/rsteps/2**0.5)) in which the field is calculated
    
    ex_XZ[y,x2] with y each index of the coordinates np.linspace(field_of_view/2,-field_of_view/2,2*int(field_of_view/rsteps/2)) and x each index for np.linspace(-field_of_view/2,field_of_view/2,2*int(field_of_view/rsteps/2)) in which the field is calculated
    
    The XZ plane is given by y=0 and the XZ plane by z=zp0 
    '''
    #XY plane: 
    Ex_XY,Ey_XY,Ez_XY=custom_mask_focus_field_XY(ex_lens,ey_lens,alpha,h,wavelength,zp0,resolution_focus,resolution_theta,resolution_phi,FOV_focus,countdown=True,x0=x0,y0=y0)
    
    #XZ plane:
    if int(resolution_z%2)==0:
        resolution_z+=1    #make the middle coordinate on Z be Z=0
        
    #passage from mm to nm and sine's law  
    focus=h/np.sin(alpha)*10**6
        
    #The Y component of incident field must be evaluated at phi-pi/2, which is equivalent to moving the rows of the matrix    
    def rotate_90º(matrix):
        a,b=np.shape(matrix)       
        aux=np.zeros((a,b),dtype=complex)        
        for i in range(a):
            aux[i-int(3*a/4),:]=matrix[i,:]
        return aux

    ey_lens=rotate_90º(ey_lens)

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
    h_theta=alpha/resolution_theta
    h_phi=2*np.pi/resolution_phi
    weight_trapezoid_rho=np.zeros(resolution_theta)+h_theta
    weight_trapezoid_rho[0]=h_theta/2
    weight_trapezoid_rho[-1]=h_theta/2
    weight_trapezoid_phi=np.zeros(resolution_phi)+h_phi
    weight_trapezoid_phi[0]=h_phi/2
    weight_trapezoid_phi[-1]=h_phi/2
    weight_trapezoid=weight_trapezoid_rho*np.vstack(weight_trapezoid_phi) #represents the area of each trapezoid for each position in phi,theta

    #define coordinates in which to calculate the field:                
    xmax=FOV_focus*2**0.5/2  #on the XZ plane the x axis is sqrt(2) times longer in order to make it analogue to the VPP and no_mask functions
    x_values=np.linspace(-xmax+x0,xmax+x0,resolution_focus)
    z_values=np.linspace(z_FOV/2,-z_FOV/2,resolution_z)
    
    Ex_XZ=np.zeros((resolution_z,resolution_focus),dtype=complex)
    Ey_XZ=np.copy(Ex_XZ)
    Ez_XZ=np.copy(Ex_XZ)
        
    #define divisions for the integration:
    theta_values=np.linspace(0,alpha,resolution_theta)    #divisions of theta in which the trapezoidal 2D integration is done
    phi_values=np.linspace(0,2*np.pi,resolution_phi)      #divisions of phi in which the trapezoidal 2D integration is done
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

    Ayx=-prefactor_y*ey_lens*weight_trapezoid
    Ayy=prefactor_x*ey_lens*weight_trapezoid
    Ayz=-prefactor_z*ey_lens*weight_trapezoid

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

