import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
from auxiliary.tmm_core import *


def interface_custom_mask_focus_field_XY(n_list,d_list,ex_lens,ey_lens,alpha,h,wavelength,z_int,zp0,resolution_x,divisions_theta,divisions_phi,x_range,countdown=True,x0=0,y0=0):
    '''
    2D integration to calculate the field focused by a high aperture lens on the XY plane with an interface
    
    Args:        
        :n_list (array): array with the refraction index of each medium of the multilayer system
        
        :d_list (array): Thickness of each interface in the multilayer system (nm), must be given as a numpy array with the first and last values a np.inf. Only used if interface=True

        :ex_lens,ey_lens (array): X and Y component of the incident field. Each position must be given in the coordinates 
        
        This arrays must have the value of the amplitude of the incident field for each value of theta and phi. Example: ex_lens[phi_position,theta_position]
        for phi_position a value in np.linspace(0,2*np.pi,divisions_phi) and theta_position a value in np.linspace(0,alpha,divisions_theta) 

        :z_int: Axial position in wich the interface is located

        :zp0: Axial position for the XY plane (given by z=zp0)
    
        :resolution_x: Number of pixels for the field at the focus, in the x and y coordinates
        
        :divisions_theta,divisions_phi: Resolution for the 2D calculus (must be the same as the size of ex_lens and ey_lens) 
    
        :wavelength: Wavelength (nm) in the vacuum 
        
        :x0, y0: Used for centering the XY field at an x0, y0 position
        
        :countdown (bool): If True, a progress bar is shown with the use of the tqdm package

        The rest of the parameters are specified in sim
    
    Returns:
        :arrays: ex,ey,ez: Cartesian components of the focused field on the XY plane, given by z=zp0
    '''


    n1=n_list[0]#first medium
    n2=n_list[-1]#last medium
    
    if countdown==True:
        print('Calculating field at the focal plane:')
        time.sleep(0.5)
        
    focus=h/np.sin(alpha)*10**6
        
    #The Y component of incident field must be evaluated at phi-pi/2, which is equivalent to moving the rows of the matrix    
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

    #Functions for computing the reflection and transmitinos coeficients between 2 interfaces. Subindexes _i meant incident and _t meand transmited  
    #given incident angle (theta_i), compute the reflection coeficients:
    n12=n1/n2 #to save computing time, might not be worth it
    '''
    #functions to integrate: Focused field without interface (Ef)
    fun1=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun2=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun3=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun4=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun5=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun6=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    
    k1=wavelength
    k2=wavelength*n21
    kz_r=zp0*2*np.pi/k1
    kz_t=zp0*2*np.pi/k2
    #functions to integrate: Reflected field (Er)
    fun1_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(-rs_i(theta)*np.sin(phi)**2+rp_i(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun2_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(rs_i(theta) + rp_i(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun3_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*rp_i(theta)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun4_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(rs_i(theta) + rp_i(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))
    fun5_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(-rs_i(theta)*np.sin(phi)**2+rp_i*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))
    fun6_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*rp_i(theta)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))

    k2=wavelength*n21
    #functions to integrate: Transmited field (Et)
    k1_minus_k2=k1-k2
    fun1_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(ts_t(theta)*np.sin(phi)**2+tp_t(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun2_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(-ts_t(theta) + tp_t(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun3_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*tp_t(theta)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun4_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(-ts_t(theta) + tp_t(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    fun5_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(ts_t(theta)*np.sin(phi)**2+tp_t(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    fun6_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*tp_t(theta)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    
    '''
    #The total field is Ef+Er for z<zp0 and Et for z>zp0
        
    #2D trapezoidal method weight:
    h_theta=alpha/divisions_theta
    h_phi=2*np.pi/divisions_phi
    weight_trapezoid_rho=np.zeros(divisions_theta)+h_theta
    weight_trapezoid_rho[0]=h_theta/2
    weight_trapezoid_rho[-1]=h_theta/2
    weight_trapezoid_phi=np.zeros(divisions_phi)+h_phi
    weight_trapezoid_phi[0]=h_phi/2
    weight_trapezoid_phi[-1]=h_phi/2
    weight_trapezoid=weight_trapezoid_rho*np.vstack(weight_trapezoid_phi)
    
    #define positions in which to calculate the field:    
    xmax=x_range/2
    x_values=np.linspace(-xmax+x0,xmax+x0,resolution_x)
    y_values=np.linspace(xmax+y0,-xmax+y0,resolution_x)
    ex=np.zeros((resolution_x,resolution_x),dtype=complex)
    ey=np.copy(ex)
    ez=np.copy(ex)
    
    
    #define divisions for the integration:
    theta_values=np.linspace(0,alpha,divisions_theta)    
    phi_values=np.linspace(0,2*np.pi,divisions_phi) 
    theta,phi=np.meshgrid(theta_values,phi_values)
    
    #now begins the integration, in order to save computing time i do the trigonometric functions separatedly and save the value into an auxiliar variable. This reduces computing time up to 8 times
    cos_theta=np.cos(theta)
    cos_theta_sqrt=cos_theta**0.5
    sin_theta=np.sin(theta)
    cos_phi=np.cos(phi)
    cos_phi_square=cos_phi**2
    sin_phi=np.sin(phi)
    sin_phi_square=sin_phi**2
    
    #For integration of the field without interphace (Ef):
    k1=2*np.pi/wavelength*n1
    k2=k1*n2/n1
    prefactor_general=cos_theta_sqrt*sin_theta*k1
    prefactor_x=prefactor_general*(cos_theta+(1-cos_theta)*sin_phi_square)
    prefactor_y=prefactor_general*(1-cos_theta)*cos_phi*sin_phi
    prefactor_z=prefactor_general*sin_theta*cos_phi
    
    Axx=-prefactor_x*ex_lens*weight_trapezoid
    Axy=-prefactor_y*ex_lens*weight_trapezoid
    Axz=-prefactor_z*ex_lens*weight_trapezoid

    Ayx=-prefactor_y*ey_lens*weight_trapezoid
    Ayy=prefactor_x*ey_lens*weight_trapezoid
    Ayz=-prefactor_z*ey_lens*weight_trapezoid
    
    #Calculus of the refraction and transmition coeficients using german's code
    rs_i_theta=np.zeros((divisions_phi,divisions_theta),dtype='complex')
    rp_i_theta=np.copy(rs_i_theta)
    ts_t_theta=np.copy(rs_i_theta)
    tp_t_theta=np.copy(rs_i_theta)
    theta_values=np.linspace(0,alpha,divisions_theta)    
    for i in range(divisions_theta):
        theta_val=theta_values[i]
        tmm_p=coh_tmm('p', n_list, d_list, theta_val, wavelength)
        tmm_s=coh_tmm('s', n_list, d_list, theta_val, wavelength)
        rs_i_theta[:,i]=tmm_s['r']
        rp_i_theta[:,i]=tmm_p['r']
        ts_t_theta[:,i]=tmm_s['t']
        tp_t_theta[:,i]=tmm_p['t']
    #debuging: auxiliar_plot_something(np.abs(rs_i_theta),0,1,0,1)
    #For integration of the reflected and transmited fields (Er and Et):
    prefactor_x_r=-prefactor_general*(-rs_i_theta*sin_phi_square+rp_i_theta*cos_phi_square*cos_theta)
    prefactor_y_r=-prefactor_general*(rs_i_theta+rp_i_theta*cos_theta)*cos_phi*sin_phi
    prefactor_z_r=prefactor_general*rp_i_theta*sin_theta*cos_phi
    
    phase_z_r=np.exp(2*1j*k1*np.cos(theta)*z_int)
    
    Axx_r=-prefactor_x_r*ex_lens*weight_trapezoid
    Axy_r=prefactor_y_r*ex_lens*weight_trapezoid
    Axz_r=prefactor_z_r*ex_lens*weight_trapezoid

    Ayx_r=phase_z_r*prefactor_y_r*ey_lens*weight_trapezoid
    Ayy_r=phase_z_r*prefactor_x_r*ey_lens*weight_trapezoid
    Ayz_r=phase_z_r*prefactor_z_r*ey_lens*weight_trapezoid
 
    #switching to complex angles in order to compute the transmited complex angles:
    theta_values_complex=np.linspace(0,alpha,divisions_theta,dtype='complex')    
    phi_values_complex=np.linspace(0,2*np.pi,divisions_phi,dtype='complex') 
    theta_complex,phi_complex=np.meshgrid(theta_values_complex,phi_values_complex)

    sin_theta_complex=np.sin(theta_complex)

    cos_theta_t=(1-(n12*sin_theta_complex)**2)**0.5
    sin_theta_t=n12*sin_theta #snell
    prefactor_general_t=(cos_theta)**0.5*sin_theta*k1

    prefactor_x_t=prefactor_general_t*(ts_t_theta*sin_phi_square+tp_t_theta*cos_phi_square*cos_theta_t)
    prefactor_y_t=prefactor_general_t*(-ts_t_theta+tp_t_theta*cos_theta_t)*cos_phi*sin_phi
    prefactor_z_t=-prefactor_general_t*tp_t_theta*sin_theta_t*cos_phi
    
    phase_z_t=np.exp(1j*z_int*(k2*cos_theta_t+k1*cos_theta))
    
    Axx_t=-phase_z_t*prefactor_x_t*ex_lens*weight_trapezoid
    Axy_t=phase_z_t*prefactor_y_t*ex_lens*weight_trapezoid
    Axz_t=phase_z_t*prefactor_z_t*ex_lens*weight_trapezoid

    Ayx_t=phase_z_t*prefactor_y_t*ey_lens*weight_trapezoid
    Ayy_t=phase_z_t*prefactor_x_t*ey_lens*weight_trapezoid
    Ayz_t=phase_z_t*prefactor_z_t*ey_lens*weight_trapezoid

    kz=zp0*k1
    kz_r=zp0*k1

    phase_kz=np.exp(1j*cos_theta*kz)
    phase_kz_r=np.exp(-1j*cos_theta*kz_r)
    kz_t=zp0*k2
    phase_kz_t=np.exp(1j*cos_theta_t*kz_t)

    if countdown==True:
        for i in tqdm(range(resolution_x)):
            x=x_values[i]
            for j,y in enumerate(y_values):#idea, rotar en phi es correr las columnas de la matriz ex, ey
                rhop=(x**2+y**2)**0.5
                phip=np.arctan2(y,x)
                kr=rhop*k1
                sin_theta_kr=sin_theta*kr
                
                phase_rho_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip)))
                phase_rho_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip)))
                
                if z_int<=zp0:
                    phase_inc_x=phase_rho_x*phase_kz_t      #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz_t      #phase for the Y incident component of the transmited field
                    ex[j,i]=np.sum(Axx_t*phase_inc_x)+np.sum(Ayx_t*phase_inc_y)
                    ey[j,i]=np.sum(Axy_t*phase_inc_x)+np.sum(Ayy_t*phase_inc_y)
                    ez[j,i]=np.sum(Axz_t*phase_inc_x)+np.sum(Ayz_t*phase_inc_y)
                else:
                    phase_inc_x=phase_rho_x*phase_kz        #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz        #phase for the Y incident component of the transmited field
                    phase_inc_x_r=phase_rho_x*phase_kz_r    #phase for the X incident component of the reflected field
                    phase_inc_y_r=phase_rho_y*phase_kz_r    #phase for the Y incident component of the reflected field
                    ex[j,i]=np.sum(Axx*phase_inc_x+Axx_r*phase_inc_x_r)+np.sum(Ayx*phase_inc_y+Ayx_r*phase_inc_y_r)
                    ey[j,i]=np.sum(Axy*phase_inc_x+Axy_r*phase_inc_x_r)+np.sum(Ayy*phase_inc_y+Ayy_r*phase_inc_y_r)
                    ez[j,i]=np.sum(Axz*phase_inc_x+Axz_r*phase_inc_x_r)+np.sum(Ayz*phase_inc_y+Ayz_r*phase_inc_y_r)
    else:
        for i,x in enumerate(x_values):
            for j,y in enumerate(y_values):
                rhop=(x**2+y**2)**0.5
                phip=np.arctan2(y,x)
                kr=rhop*k
                sin_theta_kr=sin_theta*kr
                
                phase_rho_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip)))
                phase_rho_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip)))
                
                if z_int<=zp0:
                    phase_inc_x=phase_rho_x*phase_kz_t      #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz_t      #phase for the Y incident component of the transmited field
                    ex[j,i]=np.sum(Axx_t*phase_inc_x)+np.sum(Ayx_t*phase_inc_y)
                    ey[j,i]=np.sum(Axy_t*phase_inc_x)+np.sum(Ayy_t*phase_inc_y)
                    ez[j,i]=np.sum(Axz_t*phase_inc_x)+np.sum(Ayz_t*phase_inc_y)
                else:
                    phase_inc_x=phase_rho_x*phase_kz        #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz        #phase for the Y incident component of the transmited field
                    phase_inc_x_r=phase_rho_x*phase_kz_r    #phase for the X incident component of the reflected field
                    phase_inc_y_r=phase_rho_y*phase_kz_r    #phase for the Y incident component of the reflected field
                    ex[j,i]=np.sum(Axx*phase_inc_x+Axx_r*phase_inc_x_r)+np.sum(Ayx*phase_inc_y+Ayx_r*phase_inc_y_r)
                    ey[j,i]=np.sum(Axy*phase_inc_x+Axy_r*phase_inc_x_r)+np.sum(Ayy*phase_inc_y+Ayy_r*phase_inc_y_r)
                    ez[j,i]=np.sum(Axz*phase_inc_x+Axz_r*phase_inc_x_r)+np.sum(Ayz*phase_inc_y+Ayz_r*phase_inc_y_r)
        
    ex*=-1j*focus/2/np.pi*np.exp(-1j*k1*focus)
    ey*=1j*focus/2/np.pi*np.exp(-1j*k1*focus)
    ez*=1j*focus/2/np.pi*np.exp(-1j*k1*focus)
    
    return ex,ey,ez

def interface_custom_mask_focus_field_XZ_XY(n_list,d_list,ex_lens,ey_lens,alpha,h,wavelength,z_int,z_range,resolution_z,zp0,resolution_x,divisions_theta,divisions_phi,x_range,x0=0,y0=0,z0=0,plot_plane='X'):
    '''
    2D integration to calculate the field focused by a high aperture lens on the XY and XZ planes with an interface
    
    Args:        
        :n_list (array): array with the refraction index of each medium of the multilayer system
        
        :d_list (array): Thickness of each interface in the multilayer system (nm), must be given as a numpy array with the first and last values a np.inf. Only used if interface=True

        :ex_lens,ey_lens (array): X and Y component of the incident field. Each position must be given in the coordinates 
        
        This arrays must have the value of the amplitude of the incident field for each value of theta and phi. Example: ex_lens[phi_position,theta_position]
        for phi_position a value in np.linspace(0,2*np.pi,divisions_phi) and theta_position a value in np.linspace(0,alpha,divisions_theta) 

        :z_int: Axial position in wich the interface is located

        :zp0: Axial position for the XY plane (given by z=zp0)
    
        :resolution_x: Number of pixels for the field at the focus, in the x and y coordinates
        
        :resolution_z: Number of pixels for the field at the focus, in the z coordinate
        
        :divisions_theta,divisions_phi: Resolution for the 2D calculus (must be the same as the size of ex_lens and ey_lens) 
    
        :wavelength: Wavelength (nm) in the vacuum 
        
        :x0, y0: Used for centering the XY plane at an x0, y0 position (nm)
        
        :z0: Used for centering the XZ plane at an z0 position (nm)
                
        :plot_plane (string): Available values: 'X' or 'Y', select to plot the ZX or the ZY plane respectivelly

        The rest of the parameters are specified in sim

    Returns:
        :arrays: Ex_XZ,Ey_XZ,Ez_XZ,Ex_XY,Ey_XY,Ez_XY, each one is a matrix with the amplitude of each cartesian component on the XZ plane (ex_XZ,ey_XZ,ez_XZ) or on the XY plane (ex_XY,ey_XY,ez_XY)
    
        Each index of the matrixes corresponds to a different pair of coordinates, for example: 
            
        Ex_XZ[z,x] with z each index of the coordinates np.linspace(z_range/2,-z_range/2,resolution_z) and x each index for np.linspace(-x_range/2**0.5,x_range/2**0.5,resolution_x) in which the field is calculated, the x range is sqrt(2) times longer for consistency with the VP and no_mask functions
        
        Ex_XY[y,x2] with y each index of the coordinates np.linspace(x_range/2,-x_range/2,resolution_x) and x each index for np.linspace(-x_range/2,x_range/2,resolution_x) in which the field is calculated
        
        The XZ plane is given by y=0 and the XZ plane by z=zp0 

    '''


    n1=n_list[0]
    n2=n_list[-1]
    #XY plane: 
    Ex_XY,Ey_XY,Ez_XY=interface_custom_mask_focus_field_XY(n_list,d_list,ex_lens,ey_lens,alpha,h,wavelength,z_int,zp0,resolution_x,divisions_theta,divisions_phi,x_range,True,x0,y0)
    
    #XZ plane:
    if int(resolution_z%2)==0:
        resolution_z+=1    #make the middle coordinate on Z be Z=0
        
    focus=h/np.sin(alpha)*10**6
        
    #The Y component of incident field must be evaluated at phi-pi/2, which is equivalent to moving the rows of the matrix    
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
    #functions to integrate: Focused field without interface (Ef)
    fun1=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun2=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun3=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun4=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun5=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun6=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    
    k1=2*np.pi/wavelength
    kz_r=zp0*2*np.pi/k1
    #functions to integrate: Reflected field (Er)
    fun1_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(-rs_i(theta)*np.sin(phi)**2+rp_i(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun2_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(rs_i(theta) + rp_i(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun3_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*rp_i(theta)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun4_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(rs_i(theta) + rp_i(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))
    fun5_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(-rs_i(theta)*np.sin(phi)**2+rp_i*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))
    fun6_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*rp_i(theta)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))

    #functions to integrate: Transmited field (Et), would have to be integrated for theta_t (the angle of the transmited field) wich can be complex
    k2=k1*n2/n1
    k1_minus_k2=k1-k2
    kz_t=zp0*2*np.pi/k2  
    fun1_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(ts_t(theta)*np.sin(phi)**2+tp_t(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun2_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(-ts_t(theta) + tp_t(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun3_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*tp_t(theta)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun4_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(-ts_t(theta) + tp_t(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    fun5_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)*(ts_t(theta)*np.sin(phi)**2+tp_t(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    fun6_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_int)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*tp_t(theta)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    '''

    #The total field is Ef+Er for z<zp0 and Et for z>zp0
        
    #2D trapezoidal method weight:
    h_theta=alpha/divisions_theta
    h_phi=2*np.pi/divisions_phi
    weight_trapezoid_rho=np.zeros(divisions_theta)+h_theta
    weight_trapezoid_rho[0]=h_theta/2
    weight_trapezoid_rho[-1]=h_theta/2
    weight_trapezoid_phi=np.zeros(divisions_phi)+h_phi
    weight_trapezoid_phi[0]=h_phi/2
    weight_trapezoid_phi[-1]=h_phi/2
    weight_trapezoid=weight_trapezoid_rho*np.vstack(weight_trapezoid_phi)
       
    
    #define divisions for the integration:
    theta_values=np.linspace(0,alpha,divisions_theta)    
    phi_values=np.linspace(0,2*np.pi,divisions_phi) 
    theta,phi=np.meshgrid(theta_values,phi_values)
    
    #now begins the integration, in order to save computing time i do the trigonometric functions separatedly and save the value into an auxiliar variable. This reduces computing time up to 8 times
    cos_theta=np.cos(theta)
    cos_theta_sqrt=cos_theta**0.5
    sin_theta=np.sin(theta)
    cos_phi=np.cos(phi)
    cos_phi_square=cos_phi**2
    sin_phi=np.sin(phi)
    sin_phi_square=sin_phi**2
    
    #For integration of the field without interphace (Ef):
    k1=2*np.pi/wavelength*n1
    k2=k1*n2/n1
    prefactor_general=cos_theta_sqrt*sin_theta*k1
    prefactor_x=prefactor_general*(sin_phi_square+cos_phi_square*cos_theta)
    prefactor_y=prefactor_general*(-1+cos_theta)*cos_phi*sin_phi
    prefactor_z=-prefactor_general*sin_theta*cos_phi
    
    Axx=-prefactor_x*ex_lens*weight_trapezoid
    Axy=prefactor_y*ex_lens*weight_trapezoid
    Axz=prefactor_z*ex_lens*weight_trapezoid

    Ayx=prefactor_y*ey_lens*weight_trapezoid
    Ayy=prefactor_x*ey_lens*weight_trapezoid
    Ayz=prefactor_z*ey_lens*weight_trapezoid

    #Calculus of the refraction and transmition coeficients using german's code
    rs_i_theta=np.zeros((divisions_phi,divisions_theta),dtype='complex')
    rp_i_theta=np.copy(rs_i_theta)
    ts_t_theta=np.copy(rs_i_theta)
    tp_t_theta=np.copy(rs_i_theta)
    theta_values=np.linspace(0,alpha,divisions_theta) 
    reflejado_values=np.zeros(divisions_theta,dtype='complex')
    transmitido_values=np.zeros(divisions_theta,dtype='complex')
    for i in range(divisions_theta):
        theta_val=theta_values[i]
        tmm_p=coh_tmm('p', n_list, d_list, theta_val, wavelength)
        tmm_s=coh_tmm('s', n_list, d_list, theta_val, wavelength)
        rs_i_theta[:,i]=tmm_s['r']
        rp_i_theta[:,i]=tmm_p['r']
        ts_t_theta[:,i]=tmm_s['t']
        tp_t_theta[:,i]=tmm_p['t']
        reflejado_values[i]=tmm_p['r']
        transmitido_values[i]=tmm_p['t']

    #For integration of the reflected and transmited fields (Er and Et):
    prefactor_x_r=prefactor_general*(rs_i_theta*sin_phi_square-rp_i_theta*cos_phi_square*cos_theta)
    prefactor_y_r=prefactor_general*(-rs_i_theta-rp_i_theta*cos_theta)*cos_phi*sin_phi
    prefactor_z_r=-prefactor_general*rp_i_theta*sin_theta*cos_phi
    
    phase_z_r=np.exp(2*1j*k1*np.cos(theta)*z_int)
    
    Axx_r=-prefactor_x_r*ex_lens*weight_trapezoid
    Axy_r=prefactor_y_r*ex_lens*weight_trapezoid
    Axz_r=prefactor_z_r*ex_lens*weight_trapezoid

    Ayx_r=phase_z_r*prefactor_y_r*ey_lens*weight_trapezoid
    Ayy_r=phase_z_r*prefactor_x_r*ey_lens*weight_trapezoid
    Ayz_r=phase_z_r*prefactor_z_r*ey_lens*weight_trapezoid
   
    #switching to complex angles in order to compute the transmited complex angles:
    theta_values_complex=np.linspace(0,alpha,divisions_theta,dtype='complex')    
    phi_values_complex=np.linspace(0,2*np.pi,divisions_phi,dtype='complex') 
    theta_complex,phi_complex=np.meshgrid(theta_values_complex,phi_values_complex)

    sin_theta_complex=np.sin(theta_complex)

    cos_theta_t=(1-(n1/n2*sin_theta_complex)**2)**0.5
    sin_theta_t=n1/n2*sin_theta #snell
    prefactor_general_t=(cos_theta)**0.5*sin_theta*k1

    prefactor_x_t=prefactor_general_t*(ts_t_theta*sin_phi_square+tp_t_theta*cos_phi_square*cos_theta_t)
    prefactor_y_t=prefactor_general_t*(-ts_t_theta+tp_t_theta*cos_theta_t)*cos_phi*sin_phi
    prefactor_z_t=-prefactor_general_t*tp_t_theta*sin_theta_t*cos_phi
    
    phase_z_t=np.exp(1j*z_int*(k2*cos_theta_t+k1*cos_theta))
    
    Axx_t=-phase_z_t*prefactor_x_t*ex_lens*weight_trapezoid
    Axy_t=phase_z_t*prefactor_y_t*ex_lens*weight_trapezoid
    Axz_t=phase_z_t*prefactor_z_t*ex_lens*weight_trapezoid

    Ayx_t=phase_z_t*prefactor_y_t*ey_lens*weight_trapezoid
    Ayy_t=phase_z_t*prefactor_x_t*ey_lens*weight_trapezoid
    Ayz_t=phase_z_t*prefactor_z_t*ey_lens*weight_trapezoid



    #define positions in which to calculate the field:                
    xmax=x_range/2
    x_values=np.linspace(-xmax+x0,xmax+x0,resolution_x)*(2**0.5) #on the XZ plane the x axis is calculated sqrt(2) times longer in order to make it analogue to the VPP and no_mask functions, since the maximum radial distance is larger than the maximum X or Y distance in the XY plane
    z_values=np.linspace(z_range/2+z0,-z_range/2+z0,resolution_z)
    
    Ex_XZ=np.zeros((resolution_z,resolution_x),dtype=complex)
    Ey_XZ=np.copy(Ex_XZ)
    Ez_XZ=np.copy(Ex_XZ)
        
    #define divisions for the integration:
    theta_values=np.linspace(0,alpha,divisions_theta)    
    phi_values=np.linspace(0,2*np.pi,divisions_phi) 
    theta,phi=np.meshgrid(theta_values,phi_values)
    if plot_plane=='X':
        for j in tqdm(range(resolution_z)):
            zp0=z_values[j]
            for i,x in enumerate(x_values):
                rhop=np.abs(x)
                phip=np.arctan2(0,x)
                kr=rhop*k1
                sin_theta_kr=sin_theta*kr       #because of snell's law, this factor will be the same for the reflected and transmited fields
                
                phase_rho_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip)))
                phase_rho_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip)))
                
                if z_int<=zp0:
                    kz_t=zp0*k2
                    phase_kz_t=np.exp(1j*cos_theta_t*kz_t)
                    phase_inc_x=phase_rho_x*phase_kz_t      #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz_t      #phase for the Y incident component of the transmited field
                    Ex_XZ[j,i]=np.sum(Axx_t*phase_inc_x)+np.sum(Ayx_t*phase_inc_y)
                    Ey_XZ[j,i]=np.sum(Axy_t*phase_inc_x)+np.sum(Ayy_t*phase_inc_y)
                    Ez_XZ[j,i]=np.sum(Axz_t*phase_inc_x)+np.sum(Ayz_t*phase_inc_y)
                else:
                    kz=zp0*k1
                    phase_kz=np.exp(1j*cos_theta*kz)
                    phase_kz_r=np.exp(-1j*cos_theta*kz)
                    phase_inc_x=phase_rho_x*phase_kz        #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz        #phase for the Y incident component of the transmited field
                    phase_inc_x_r=phase_rho_x*phase_kz_r    #phase for the X incident component of the reflected field
                    phase_inc_y_r=phase_rho_y*phase_kz_r    #phase for the Y incident component of the reflected field
                    Ex_XZ[j,i]=np.sum(Axx*phase_inc_x+Axx_r*phase_inc_x_r)+np.sum(Ayx*phase_inc_y+Ayx_r*phase_inc_y_r)
                    Ey_XZ[j,i]=np.sum(Axy*phase_inc_x+Axy_r*phase_inc_x_r)+np.sum(Ayy*phase_inc_y+Ayy_r*phase_inc_y_r)
                    Ez_XZ[j,i]=np.sum(Axz*phase_inc_x+Axz_r*phase_inc_x_r)+np.sum(Ayz*phase_inc_y+Ayz_r*phase_inc_y_r)
    else:
        if plot_plane=='Y':           
            for j in tqdm(range(resolution_z)):
                zp0=z_values[j]
                for i,y in enumerate(x_values):
                    rhop=np.abs(y)
                    phip=np.arctan2(y,0)
                    kr=rhop*k1
                    sin_theta_kr=sin_theta*kr       #because of snell's law, this factor will be the same for the reflected and transmited fields
                    
                    phase_rho_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip)))
                    phase_rho_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip)))
                    
                    if z_int<=zp0:
                        kz_t=zp0*k2
                        phase_kz_t=np.exp(1j*cos_theta_t*kz_t)
                        phase_inc_x=phase_rho_x*phase_kz_t      #phase for the X incident component of the transmited field
                        phase_inc_y=phase_rho_y*phase_kz_t      #phase for the Y incident component of the transmited field
                        Ex_XZ[j,i]=np.sum(Axx_t*phase_inc_x)+np.sum(Ayx_t*phase_inc_y)
                        Ey_XZ[j,i]=np.sum(Axy_t*phase_inc_x)+np.sum(Ayy_t*phase_inc_y)
                        Ez_XZ[j,i]=np.sum(Axz_t*phase_inc_x)+np.sum(Ayz_t*phase_inc_y)
                    else:
                        kz=zp0*k1
                        phase_kz=np.exp(1j*cos_theta*kz)
                        phase_kz_r=np.exp(-1j*cos_theta*kz)
                        phase_inc_x=phase_rho_x*phase_kz        #phase for the X incident component of the transmited field
                        phase_inc_y=phase_rho_y*phase_kz        #phase for the Y incident component of the transmited field
                        phase_inc_x_r=phase_rho_x*phase_kz_r    #phase for the X incident component of the reflected field
                        phase_inc_y_r=phase_rho_y*phase_kz_r    #phase for the Y incident component of the reflected field
                        Ex_XZ[j,i]=np.sum(Axx*phase_inc_x+Axx_r*phase_inc_x_r)+np.sum(Ayx*phase_inc_y+Ayx_r*phase_inc_y_r)
                        Ey_XZ[j,i]=np.sum(Axy*phase_inc_x+Axy_r*phase_inc_x_r)+np.sum(Ayy*phase_inc_y+Ayy_r*phase_inc_y_r)
                        Ez_XZ[j,i]=np.sum(Axz*phase_inc_x+Axz_r*phase_inc_x_r)+np.sum(Ayz*phase_inc_y+Ayz_r*phase_inc_y_r)
        else:
            print('Options for plot_plane are \'X\' and \'Y\' ')
    Ex_XZ*=1j*focus*np.exp(1j*k1*focus)/2/np.pi
    Ey_XZ*=1j*focus*np.exp(1j*k1*focus)/2/np.pi
    Ez_XZ*=1j*focus*np.exp(1j*k1*focus)/2/np.pi
    
    # plot coeficiente de reflexion
    # plt.rcParams['font.size']=14
    # fig = plt.figure(figsize=(8, 8))
    # spec = fig.add_gridspec(ncols=1, nrows=2)

    # ax2 = fig.add_subplot(spec[1, 0])
    # ax2.plot(np.rad2deg(theta_values),np.angle(reflejado_values,deg=True))
    # ax2.set_xlabel('\u03B8 (º)')
    # ax2.set_ylabel('Fase en la reflectividad')
    
    # ax1 = fig.add_subplot(spec[0, 0],sharex=ax2)
    # ax1.set_title('Coeficiente de reflección')
    # ax1.plot(np.rad2deg(theta_values),np.abs(reflejado_values))
    # ax1.set_ylabel('Módulo de reflectividad')
    
    
    return Ex_XZ,Ey_XZ,Ez_XZ,Ex_XY,Ey_XY,Ez_XY

