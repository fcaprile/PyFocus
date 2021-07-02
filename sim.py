'''
High-level functions to simulate the foci obtained when: Focusing a gaussian beam, focusing a gaussian beam modulated with a VP mask and focusing a gaussian beam modulated with a custom phase mask

The functions use the 'parameters' array, defined as:
    
    parameters=np.array((NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,ds,zint,figure_name), dtype=object)
    
With:
    
        NA (float): Numerical aperture
        
        n (float or array, depends on interface): Refraction index for the medium of the optical system. If interface=True it must be given as a numpy array
        
        h (float): Radius of aperture of the objective lens(mm)
        
        f (float): Focal distance of the objective lens (mm)
        
        w0 (float): Radius of the incident gaussian beam (mm)
        
        wavelength (float): Wavelength in the medium of the objective lens (equals to wavelength in vacuum/n)
        
        gamma (float): Parameter that determines the polarization, arctan(ey/ex) (gamma=45, beta=90 gives right circular polarization and a donut shape)
        
        beta (float): parameter that determines the polarization, phase difference between ex and ey (gamma=45, beta=90 gives right circular polarization and a donut shape)
        
        zp0 (float): Axial position in which to calculate the XY plane, given by z=zp0, (nm)
        
        rsteps (float): Resolution in the X or Y coordinate for the focused field (nm)
        
        zsteps (float): Resolution in the axial coordinate (Z) for the focused field (nm)
        
        field_of_view (float): Field of view in the X or Y coordinate in which the focused field is calculated (nm)
        
        z_field_of_view (float): field of view in the axial coordinate (z) in which the focused field is calculated (nm)
        
        I_0 (float): Incident field intensity (uW/cm^2)
        
        L (float, optional): Distance between phase mask and objective lens (mm), only used if propagation=True
        
        R (float, optional): Phase mask radius (mm), only used if propagation=True
        
        ds (array, optional): Thickness of each interface in the multilayer system (nm), must be given as a numpy array with the first and last values a np.inf. Only used if interface=True
        
        zint (float, optional): Axial position of the interphase. Only used if interface=True
        
        figure_name (string, optional): Name for the images of the field. Also used as saving name if using the UI    
'''

import numpy as np
import time


from auxiliary.VP_functions import VP_integration, VP_fields, VP_fraunhofer
from auxiliary.no_mask_functions import no_mask_integration, no_mask_fields
from auxiliary.custom_mask_functions import generate_incident_field, plot_in_cartesian, custom_mask_objective_field, custom_mask_focus_field_XZ_XY
from auxiliary.interface import interface_custom_mask_focus_field_XZ_XY

def VP(propagation=False,interface=False,NA=1.4,n=1.5,h=3,f=3.21,w0=5,wavelength=640,gamma=45,beta=90,zp0=0,rsteps=5,zsteps=8,field_of_view=1000,z_field_of_view=2000,I0=1,L='',R='',ds='',zint='',figure_name=''):#f (the focal distance) is given by the sine's law, but can be modified if desired
    '''
    Simulate the field obtained by focusing a gaussian beam modulated by a VP mask 
    
    Args:
        
        propagation (bool): True for calculating and ploting the field incident on the lens by fraunhofer's difractin formula, in which case R and L are needed; False for calculating the field incident on the lens while neglecting the propagation
        
        interface (bool): True for calculating the focused field with an interface, in which case ds and z_int are needed; Flase for calculating the field obtained without an interface
    
        NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,ds,zint,figure_name: Simulation parameters, defined as part of the 'parameters' array
        
    Returns:
        
        arrays: ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY, each one is a matrix with the amplitude of each cartesian component on the XZ plane (ex_XZ,ey_XZ,ez_XZ) or on the XY plane (ex_XY,ey_XY,ez_XY)
    
    Each index of the matrixes corresponds to a different pair of coordinates, for example: 
        
    ex_XZ[z,x] with z each index of the coordinates np.linspace(z_field_of_view/2,-z_field_of_view/2,2*int(z_field_of_view/zsteps/2)) and x each index for np.linspace(-field_of_view/2**0.5,field_of_view/2**0.5,2*int(field_of_view/rsteps/2**0.5)) in which the field is calculated
    
    ex_XZ[y,x2] with y each index of the coordinates np.linspace(field_of_view/2,-field_of_view/2,2*int(field_of_view/rsteps/2)) and x each index for np.linspace(-field_of_view/2,field_of_view/2,2*int(field_of_view/rsteps/2)) in which the field is calculated
    
    The XZ plane is given by y=0 and the XZ plane by z=zp0 
    
    The radial field of view in the XZ plane is sqrt(2) times bigger to allow a square field of view for the XY plane (the maximum radial position is higher than the maximum x or y position)
    '''
    
    
    if interface==False:#no interface
        alpha=np.arcsin(NA / n)
        if propagation==False:
            II1,II2,II3,II4,II5=VP_integration(alpha,n,f,w0,wavelength/n,rsteps,zsteps,field_of_view,z_field_of_view)
            ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=VP_fields(II1,II2,II3,II4,II5,wavelength/n,I0,gamma,beta,rsteps,zsteps,field_of_view,z_field_of_view,0,n,f,zp0)                        
        else:
            E_rho,Ex,Ey=VP_fraunhofer(gamma,beta,1000,R,L,I0,wavelength/n,h,w0,2000,20,figure_name=figure_name)
            '''
            Funny enough, doing a 2D integration with the trapezoid method will be faster than the 1D integration, since the function converges reeeeally slowly with scipy.integrate.quad, sorry if this is confusing. 1D integration method is used by the funcions "VP_integration_with_propagation" and "VP_fields_with_propagation" in VP.py, but unimplemented here
            '''
            #Usual parameters for integration precision:
            #resolution for field at objective
            resolution_theta=200
            resolution_phi=200

            ex_lens=np.zeros((resolution_phi,resolution_theta),dtype=complex)
            ey_lens=np.zeros((resolution_phi,resolution_theta),dtype=complex)
        
            theta_values=np.linspace(0,alpha,resolution_theta)  #divisions of theta in which the trapezoidal 2D integration is done
            rhop_values=np.sin(theta_values)*f              #given by the sine's law
            phip_values=np.linspace(0,2*np.pi,resolution_phi)   #divisions of phi in which the trapezoidal 2D integration is done
            for i,phip in enumerate(phip_values):
                for j,rhop in enumerate(rhop_values):
                    ex_lens[i,j]=E_rho(rhop)*np.exp(1j*phip)
                    ey_lens[i,j]=E_rho(rhop)*np.exp(1j*phip)
            ex_lens*=np.cos(gamma*np.pi/180)*I0**0.5
            ey_lens*=np.sin(gamma*np.pi/180)*np.exp(1j*beta*np.pi/180)*I0**0.5

            
            #resolution for field near the focus
            resolution_focus=int(field_of_view/rsteps)
            resolution_z=int(z_field_of_view/zsteps)
            
            #smaller numbers give faster integration times
            
            #calculate field at the focal plane:
            ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,wavelength/n,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view)
    
    else:#interface is given
        alpha=np.arcsin(NA / n[0])
        mask_function_VP=lambda theta, phi,w0,f,wavelength: np.exp(1j*phi)# is the VP mask function
        #Usual parameters for integration precision:
        #resolution for field at objective
        resolution_theta=200
        resolution_phi=200
        
        #resolution for field near the focus
        resolution_focus=int(field_of_view/rsteps)
        resolution_z=int(z_field_of_view/zsteps)
        
        #smaller numbers give faster integration times
        
        if propagation==False:
            #calculate field inciding on the lens by multiplying the phase mask's function and the gaussian amplitude
            #(ex_lens,ey_lens) are 2 matrixes with the values of the incident amplitude for each value in phi,theta                   
            ex_lens,ey_lens=generate_incident_field(mask_function_VP,alpha,f,resolution_phi,resolution_theta,gamma,beta,w0,I0,wavelength/n[0])            
            #plot field at the entrance of the lens:
            #since ex_lens and ey_lens are given in theta and phi coordinates, the have to be transformed to cartesian in order to be ploted, hence the name of this function
        else:
            #calculate field inciding on the lens by fraunhofer's difraction formula
            #N_rho and N_phi are the number of divisions for fraunhoffer's integral by the trapecium's method
            N_rho=400
            N_phi=400                
            ex_lens,ey_lens,I_cartesian,Ex_cartesian,Ey_cartesian=custom_mask_objective_field(h,gamma,beta,resolution_theta,resolution_phi,N_rho,N_phi,alpha,f,mask_function_VP,R,L,I0,wavelength*10**-6,w0,figure_name,plot=True)

        #calculate field at the focal plane:
        ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=interface_custom_mask_focus_field_XZ_XY(n,ds,ex_lens,ey_lens,alpha,h,wavelength,zint,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view)
    
    return ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY

def no_mask(propagation=False,interface=False,NA=1.4,n=1.5,h=3,f=3.21,w0=5,wavelength=640,gamma=45,beta=90,zp0=0,rsteps=5,zsteps=8,field_of_view=1000,z_field_of_view=2000,I0=1,L='',R='',ds='',zint='',figure_name=''):#f (the focal distance) is given by the sine's law, but can be modified if desired
    '''
    Simulate the field obtained by focusing a gaussian beam without being modulated in phase
    
    Args:
        
        propagation (bool): Kept for homogeneity of the functions, True yields no diference from False, where the field incident on the lens is calculated by neglecting the propagation towards the objective lens
        
        interface (bool): True for calculating the focused field with an interface, in which case ds and z_int are needed; Flase for calculating the field obtained without an interface
    
        NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,ds,zint,figure_name: Simulation parameters, defined as part of the 'parameters' array
        
    Returns:
        
        arrays: ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY, each one is a matrix with the amplitude of each cartesian component on the XZ plane (ex_XZ,ey_XZ,ez_XZ) or on the XY plane (ex_XY,ey_XY,ez_XY)
    
    Each index of the matrixes corresponds to a different pair of coordinates, for example: 
        
    ex_XZ[z,x] with z each index of the coordinates np.linspace(z_field_of_view/2,-z_field_of_view/2,2*int(z_field_of_view/zsteps/2)) and x each index for np.linspace(-field_of_view/2**0.5,field_of_view/2**0.5,2*int(field_of_view/rsteps/2**0.5)) in which the field is calculated
    
    ex_XZ[y,x2] with y each index of the coordinates np.linspace(field_of_view/2,-field_of_view/2,2*int(field_of_view/rsteps/2)) and x each index for np.linspace(-field_of_view/2,field_of_view/2,2*int(field_of_view/rsteps/2)) in which the field is calculated
    
    The XZ plane is given by y=0 and the XZ plane by z=zp0 
    
    The radial field of view in the XZ plane is sqrt(2) times bigger to allow a square field of view for the XY plane (the maximum radial position is higher than the maximum x or y position)
    '''
    
    if propagation==True:
        print('Propagation for field incident on the lens is only done for VP and custom masks. Calculating focused field with propagation=False:')
        time.sleep(0.5)
    if interface==False:#no interface
        alpha=np.arcsin(NA / n)
        II1,II2,II3=no_mask_integration(alpha,n,f,w0,wavelength/n,field_of_view,z_field_of_view,zsteps,rsteps)
        ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=no_mask_fields(II1,II2,II3,wavelength/n,I0,beta,gamma,zsteps,rsteps,field_of_view,z_field_of_view,0,n,f,zp0)

    else:#interface
        alpha=np.arcsin(NA / n[0])
        mask_function=lambda theta, phi,w0,f,wavelength: 1# is the nule mask function
        resolution_theta=200
        resolution_phi=200
        resolution_focus=int(field_of_view/rsteps)
        resolution_z=int(z_field_of_view/zsteps)

        ex_lens,ey_lens=generate_incident_field(mask_function,alpha,f,resolution_phi,resolution_theta,gamma,beta,w0,I0,wavelength/n[0])
        ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=interface_custom_mask_focus_field_XZ_XY(n,ds,ex_lens,ey_lens,alpha,h,wavelength,zint,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view,x0=0,y0=0,z0=0,plot_plane='X')

    return ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY

def custom(mask_function, propagation=False,interface=False,NA=1.4,n=1.5,h=3,f=3.21,w0=5,wavelength=640,gamma=45,beta=90,zp0=0,rsteps=5,zsteps=8,field_of_view=1000,z_field_of_view=2000,I0=1,L='',R='',ds='',zint='',figure_name='',resolution_theta=200,resolution_phi=200):#f (the focal distance) is given by the sine's law, but can be modified if desired
    '''
    Simulate the field obtained by focusing a gaussian beam modulated by a custom phase mask
    
    The amplitude term of a gaussian beam is already added to the incident field despite the phase mask used, if this is not desired w0=100 (a big number) makes this term essentially 1

    Args:
        
        propagation (bool): True for calculating and ploting the field incident on the lens by fraunhofer's difractin formula, in which case R and L are needed; False for calculating the field incident on the lens while neglecting the propagation
        
        interface (bool): True for calculating the focused field with an interface, in which case ds and z_int are needed; Flase for calculating the field obtained without an interface
    
        NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,ds,zint,figure_name: Simulation parameters, defined as part of the 'parameters' array
        
    Returns:
        
        arrays: ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY, each one is a matrix with the amplitude of each cartesian component on the XZ plane (ex_XZ,ey_XZ,ez_XZ) or on the XY plane (ex_XY,ey_XY,ez_XY)
    
    Each index of the matrixes corresponds to a different pair of coordinates, for example: 
        
    ex_XZ[z,x] with z each index of the coordinates np.linspace(z_field_of_view/2,-z_field_of_view/2,2*int(z_field_of_view/zsteps/2)) and x each index for np.linspace(-field_of_view/2**0.5,field_of_view/2**0.5,2*int(field_of_view/rsteps/2**0.5)) in which the field is calculated
    
    ex_XZ[y,x2] with y each index of the coordinates np.linspace(field_of_view/2,-field_of_view/2,2*int(field_of_view/rsteps/2)) and x each index for np.linspace(-field_of_view/2,field_of_view/2,2*int(field_of_view/rsteps/2)) in which the field is calculated
    
    The XZ plane is given by y=0 and the XZ plane by z=zp0 
    
    The radial field of view in the XZ plane is sqrt(2) times bigger to allow a square field of view for the XY plane (the maximum radial position is higher than the maximum x or y position)
    
    '''
    #moving to wavelength in the optical system's medium
    if interface==False:
        alpha=np.arcsin(NA / n)
        wavelength/=n
    else:
        alpha=np.arcsin(NA / n[0])
        wavelength/=n[0]

    # mask_function=lambda theta, phi: np.exp(1j*phi) is the VP mask function
    #Usual parameters for integration precision:
    
    #resolution for field near the focus
    resolution_focus=int(field_of_view/rsteps)
    resolution_z=int(z_field_of_view/zsteps)
    
    #smaller numbers give faster integration times
    
    #pasage to mm:
    wavelength*=10**-6
    
    if callable(mask_function)==True:#if a function is given, turn it into the ex_lens and ey_lens variables
        #resolution for field at objective
        if propagation==False:
            #calculate field inciding on the lens by multiplying the phase mask's function and the gaussian amplitude
            #(ex_lens,ey_lens) are 2 matrixes with the values of the incident amplitude for each value in phi,theta                   
            ex_lens,ey_lens=generate_incident_field(mask_function,alpha,f,resolution_phi,resolution_theta,gamma,beta,w0,I0,wavelength)
            #plot field at the entrance of the lens:
            #since ex_lens and ey_lens are given in theta and phi coordinates, the have to be transformed to cartesian in order to be ploted, hence the name of this function
            plot_in_cartesian(ex_lens,ey_lens,h,alpha,f,figure_name)
        else:
            #calculate field inciding on the lens by fraunhofer's difraction formula
            #N_rho and N_phi are the number of divisions for fraunhoffer's integral by the trapecium's method
            N_rho=500
            N_phi=500                
            ex_lens,ey_lens,I_cartesian,Ex_cartesian,Ey_cartesian=custom_mask_objective_field(h,gamma,beta,resolution_theta,resolution_phi,N_rho,N_phi,alpha,f,mask_function,R,L,I0,wavelength,w0,figure_name,plot=True)
            plot_in_cartesian(ex_lens,ey_lens,h,alpha,f,figure_name)

    elif type(mask_function)==np.ndarray:#if an array is given
        ex_lens,ey_lens=mask_function*np.cos(gamma*np.pi/180)*I0**0.5,mask_function*np.sin(gamma*np.pi/180)*np.exp(1j*beta*np.pi/180)*I0**0.5 #make the x and y component based on polarization
        plot_in_cartesian(ex_lens,ey_lens,h,alpha,f,figure_name)
        resolution_phi,resolution_theta=np.shape(mask_function)
    else:
        print('Wrong format for mask function, acceptable are functions or arrays')
        
    #calculate field at the focal plane:
    #pasage to nm:
    wavelength*=10**6

    if interface==False:#no interface
        ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,wavelength,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view)
    else:
        wavelength*=n[0] #to return to wavelength at vacuum
        ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=interface_custom_mask_focus_field_XZ_XY(n,ds,ex_lens,ey_lens,alpha,h,wavelength,zint,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view)

    return ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY

