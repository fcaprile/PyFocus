'''
High-level functions to simulate the foci obtained when: Focusing a gaussian beam, focusing a gaussian beam modulated with a VP mask and focusing a gaussian beam modulated with a custom phase mask

The functions use the 'parameters' array, defined as:
    
    parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)
    
With:
    
        :NA (float): Numerical aperture
        
        :n (float or array): Type depends on presence of multilayer: float for no multilayer, array for multilayer. Refraction index for the medium of the optical system.
        
        :h (float): Radius of aperture of the objective lens(mm)
                
        :w0 (float): Radius of the incident gaussian beam (mm)
        
        :wavelength (float): Wavelength in the medium of the objective lens (equals to wavelength in vacuum/n)
        
        :gamma (float): Parameter that determines the polarization, arctan(ey/ex) (gamma=45, beta=90 gives right circular polarization and a donut shape)
        
        :beta (float): parameter that determines the polarization, phase difference between ex and ey (gamma=45, beta=90 gives right circular polarization and a donut shape)
        
        :z (float): Axial position for the XY plane (nm)
        
        :x_steps (float): Resolution in the X or Y coordinate for the focused field (nm)
        
        :z_steps (float): Resolution in the axial coordinate (Z) for the focused field (nm)
        
        :x_range (float): Field of view in the X or Y coordinate in which the focused field is calculated (nm)
        
        :z_range (float): field of view in the axial coordinate (z) in which the focused field is calculated (nm)
        
        :I_0 (float): Incident field intensity (mW/cm^2)
        
        :L (float, optional): Distance between phase mask and objective lens (mm), only used if propagation=True
        
        :R (float, optional): Phase mask radius (mm), only used if propagation=True
        
        :ds (array, optional): Thickness of each interface in the multilayer system (nm), must be given as a numpy array with the first and last values a np.inf. Only used if multilayer=True
        
        :z_int (float, optional): Axial position of the interphase. Only used if multilayer=True
        
        :figure_name (string, optional): Name for the images of the field. Also used as saving name if using the UI    
'''

import numpy as np
import time

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from PyFocus.VP_functions import VP_integration, VP_fields, VP_fraunhofer
from PyFocus.no_mask_functions import no_mask_integration, no_mask_fields
from PyFocus.custom_mask_functions import generate_incident_field, plot_in_cartesian, custom_mask_objective_field, custom_mask_focus_field_XZ_XY
from PyFocus.interface import interface_custom_mask_focus_field_XZ_XY

def VP(propagation=False,multilayer=False,NA=1.4,n=1.5,h=3,w0=5,wavelength=640,gamma=45,beta=90,z=0,x_steps=5,z_steps=8,x_range=1000,z_range=2000,I0=1,L='',R='',ds='',z_int='',figure_name=''):
    '''
    Simulate the field obtained by focusing a gaussian beam modulated by a VP mask 
    
    Args:        
        :propagation (bool): True for calculating and ploting the field incident on the lens by fraunhofer's difractin formula, in which case R and L are needed; False for calculating the field incident on the lens while neglecting the propagation 
       
        :multilayer (bool): True for calculating the focused field with an multilayer, in which case ds and z_int are needed; Flase for calculating the field obtained without an multilayer
    
        :Parameters: NA,n,h,w0,wavelength,gamma,beta,z,x_steps,z_steps,x_range,z_range,I0,L,R,ds,z_int,figure_name: Simulation parameters, defined as part of the 'parameters' array
        
    Returns:        
        :arrays: ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY, each one is a matrix with the amplitude of each cartesian component on the XZ plane (ex_XZ,ey_XZ,ez_XZ) or on the XY plane (ex_XY,ey_XY,ez_XY)
    
        Each index of the matrixes corresponds to a different pair of coordinates, for example: 
            
        ex_XZ[z,x] with z each index of the coordinates np.linspace(z_range/2,-z_range/2,2*int(z_range/z_steps/2)) and x each index for np.linspace(-x_range/2**0.5,x_range/2**0.5,2*int(x_range/x_steps/2**0.5)) in which the field is calculated. 
        
        IMPORTANT: It is worth noting the x positions are sqrt(2) times higher to allow a square field of view for the XY plane (because the maximum radial position to be calculated for the XY plane is higher than the maximum x or y position)
        
        ex_XZ[y,x2] with y each index of the coordinates np.linspace(x_range/2,-x_range/2,2*int(x_range/x_steps/2)) and x each index for np.linspace(-x_range/2,x_range/2,2*int(x_range/x_steps/2)) in which the field is calculated
        
        The XZ plane is given by y=0 and the XY plane by the z parameter
        
        The unit of the intensity is the same as the initial intensity (mW/cm^2)
    '''
    
    
    if multilayer==False:#no multilayer
        alpha=np.arcsin(NA / n)
        f=h*n/NA#sine's law
        if propagation==False:
            II1,II2,II3,II4,II5=VP_integration(alpha,n,f,w0,wavelength/n,x_steps,z_steps,x_range,z_range)
            ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=VP_fields(II1,II2,II3,II4,II5,wavelength/n,I0,gamma,beta,x_steps,z_steps,x_range,z_range,0,n,f,z)                        
        else:
            E_rho,Ex,Ey=VP_fraunhofer(gamma,beta,1000,R,L,I0,wavelength/n,2*h,w0,2000,20,plot=True,figure_name=figure_name)
            '''
            To calculate the focused field, doing a 2D integration with the trapezoid method will be faster than the 1D integration, since the function converges really slowly with scipy.integrate.quad, requiring an hour at least. 1D integration method is used by the funcions "VP_integration_with_propagation" and "VP_fields_with_propagation" in VP.py, but unimplemented here
            '''
            #Usual parameters for integration precision:
            #resolution for field at objective
            divisions_theta=200
            divisions_phi=200

            ex_lens=np.zeros((divisions_phi,divisions_theta),dtype=complex)
            ey_lens=np.zeros((divisions_phi,divisions_theta),dtype=complex)
        
            theta_values=np.linspace(0,alpha,divisions_theta)  #divisions of theta in which the trapezoidal 2D integration is done
            rhop_values=np.sin(theta_values)*f              #given by the sine's law
            phip_values=np.linspace(0,2*np.pi,divisions_phi)   #divisions of phi in which the trapezoidal 2D integration is done
            for i,phip in enumerate(phip_values):
                for j,rhop in enumerate(rhop_values):
                    ex_lens[i,j]=E_rho(rhop)*np.exp(1j*phip)
                    ey_lens[i,j]=E_rho(rhop)*np.exp(1j*phip)
            ex_lens*=np.cos(gamma*np.pi/180)*I0**0.5
            ey_lens*=np.sin(gamma*np.pi/180)*np.exp(1j*beta*np.pi/180)*I0**0.5

            #resolution for field near the focus, smaller numbers give faster integration times
            resolution_focus=int(x_range/x_steps)
            resolution_z=int(z_range/z_steps)
            
            #calculate field at the focal plane:
            ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,wavelength/n,z_range,resolution_z,z,resolution_focus,divisions_theta,divisions_phi,x_range)
    
    else:#multilayer is given
        alpha=np.arcsin(NA / n[0])
        f=h*n[0]/NA    
        custom_field_function_VP=lambda theta, phi,w0,f,wavelength: np.exp(1j*phi)# is the VP mask function
        #Usual parameters for integration precision:
        #resolution for field at objective
        divisions_theta=200
        divisions_phi=200
        
        #resolution for field near the focus
        resolution_focus=int(x_range/x_steps)
        resolution_z=int(z_range/z_steps)
        
        #smaller numbers give faster integration times
        
        if propagation==False:
            #calculate field inciding on the lens by multiplying the phase mask's function and the gaussian amplitude
            #(ex_lens,ey_lens) are 2 matrixes with the values of the incident amplitude for each value in phi,theta                   
            ex_lens,ey_lens=generate_incident_field(custom_field_function_VP,alpha,f,divisions_phi,divisions_theta,gamma,beta,w0,I0,wavelength/n[0])            
            #plot field at the entrance of the lens:
            #since ex_lens and ey_lens are given in theta and phi coordinates, the have to be transformed to cartesian in order to be ploted, hence the name of this function
        else:
            #calculate field inciding on the lens by fraunhofer's difraction formula
            #N_rho and N_phi are the number of divisions for fraunhoffer's integral by the trapecium's method
            N_rho=400
            N_phi=400                
            ex_lens,ey_lens,I_cartesian,Ex_cartesian,Ey_cartesian=custom_mask_objective_field(h,gamma,beta,divisions_theta,divisions_phi,N_rho,N_phi,alpha,f,custom_field_function_VP,R,L,I0,wavelength,w0,figure_name,plot=True)

        #calculate field at the focal plane:
        ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=interface_custom_mask_focus_field_XZ_XY(n,ds,ex_lens,ey_lens,alpha,h,wavelength,z_int,z_range,resolution_z,z,resolution_focus,divisions_theta,divisions_phi,x_range)
    
    return ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY

def no_mask(propagation=False,multilayer=False,NA=1.4,n=1.5,h=3,w0=5,wavelength=640,gamma=45,beta=90,z=0,x_steps=5,z_steps=8,x_range=1000,z_range=2000,I0=1,L='',R='',ds='',z_int='',figure_name=''):#f (the focal distance) is given by the sine's law, but can be modified if desired
    '''
    Simulate the field obtained by focusing a gaussian beam without being modulated in phase
    
    Args:        
        :propagation (bool): Kept for homogeneity of the functions, True yields no diference from False, where the field incident on the lens is calculated by neglecting the propagation towards the objective lens
        
        :multilayer (bool): True for calculating the focused field with an multilayer, in which case ds and z_int are needed; Flase for calculating the field obtained without an multilayer
    
        :Parameters: NA,n,h,w0,wavelength,gamma,beta,z,x_steps,z_steps,x_range,z_range,I0,L,R,ds,z_int,figure_name: Simulation parameters, defined as part of the 'parameters' array
        
    Returns:        
        :arrays: ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY, each one is a matrix with the amplitude of each cartesian component on the XZ plane (ex_XZ,ey_XZ,ez_XZ) or on the XY plane (ex_XY,ey_XY,ez_XY)
    
        Each index of the matrixes corresponds to a different pair of coordinates, for example: 
            
        ex_XZ[z,x] with z each index of the coordinates np.linspace(z_range/2,-z_range/2,2*int(z_range/z_steps/2)) and x each index for np.linspace(-x_range/2**0.5,x_range/2**0.5,2*int(x_range/x_steps/2**0.5)) in which the field is calculated. 
        
        IMPORTANT: It is worth noting the x positions are sqrt(2) times higher to allow a square field of view for the XY plane (because the maximum radial position to be calculated for the XY plane is higher than the maximum x or y position)
        
        ex_XZ[y,x2] with y each index of the coordinates np.linspace(x_range/2,-x_range/2,2*int(x_range/x_steps/2)) and x each index for np.linspace(-x_range/2,x_range/2,2*int(x_range/x_steps/2)) in which the field is calculated
        
        The XZ plane is given by y=0 and the XY plane by the z parameter 
        
        The unit of the intensity is the same as the initial intensity (mW/cm^2)
    '''
    
    if propagation==True:
        print('Propagation for field incident on the lens is only done for VP and custom masks. Calculating focused field with propagation=False:')
        time.sleep(0.5)
    if multilayer==False:#no multilayer
        alpha=np.arcsin(NA / n)
        f=h*n/NA
        II1,II2,II3=no_mask_integration(alpha,n,f,w0,wavelength/n,x_range,z_range,z_steps,x_steps)
        ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=no_mask_fields(II1,II2,II3,wavelength/n,I0,beta,gamma,z_steps,x_steps,x_range,z_range,0,n,f,z)

    else:#multilayer
        f=h*n[0]/NA    
        alpha=np.arcsin(NA / n[0])
        custom_field_function=lambda theta, phi,w0,f,wavelength: 1# is the nule mask function
        divisions_theta=200
        divisions_phi=200
        resolution_focus=int(x_range/x_steps)
        resolution_z=int(z_range/z_steps)

        ex_lens,ey_lens=generate_incident_field(custom_field_function,alpha,f,divisions_phi,divisions_theta,gamma,beta,w0,I0,wavelength/n[0])
        ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=interface_custom_mask_focus_field_XZ_XY(n,ds,ex_lens,ey_lens,alpha,h,wavelength,z_int,z_range,resolution_z,z,resolution_focus,divisions_theta,divisions_phi,x_range,x0=0,y0=0,z0=0,plot_plane='X')

    return ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY

def custom(custom_field_function, propagation=False,multilayer=False,NA=1.4,n=1.5,h=3,w0=5,wavelength=640,gamma=45,beta=90,z=0,x_steps=5,z_steps=8,x_range=1000,z_range=2000,I0=1,L='',R='',ds='',z_int='',figure_name='',divisions_theta=200,divisions_phi=200,plot_Ei=True):#f (the focal distance) is given by the sine's law, but can be modified if desired
    '''
    Simulate the field obtained by focusing a gaussian beam modulated by a custom phase mask
    
    Args:        
        :custom_field_function (array or function): Parameter defining the custom phase mask to be simulated:
            If array: Complex amplitude of the custom phase mask for each value of the spherical coordinates theta and phi: custom_mask[phi_position,theta_position] for phi_position a value in np.linspace(0,2*np.pi,divisions_phi) and theta_position a value in np.linspace(0,alpha,divisions_theta) 
                
            If function: Analytical function that defines the phase mask, must be a function of the 5 internal variables: rho, phi, w0, f and k, with:
                
            rho: Radial coordinate from 0 to the aperture radius of the objective.
            
            phi: Azimutal coordinate from 0 to 2pi.
            
            w0: Radius of the incident gaussian beam.
            
            f: Focal distane of the objective lens
            
            k: Wavenumber in the objective lens medium IMPORTANT:(mm)
            
        The real part defines the amplitude of the incident field
        
        :propagation (bool): True for calculating and ploting the field incident on the lens by fraunhofer's difractin formula, in which case R and L are needed; False for calculating the field incident on the lens while neglecting the propagation
        
        :multilayer (bool): True for calculating the focused field with an multilayer, in which case ds and z_int are needed; Flase for calculating the field obtained without an multilayer
    
        :Parameters: NA,n,h,w0,wavelength,gamma,beta,z,x_steps,z_steps,x_range,z_range,I0,L,R,ds,z_int,figure_name: Simulation parameters, defined as part of the 'parameters' array
    Returns:        
        :arrays: ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY, each one is a matrix with the amplitude of each cartesian component on the XZ plane (ex_XZ,ey_XZ,ez_XZ) or on the XY plane (ex_XY,ey_XY,ez_XY)
    
        Each index of the matrixes corresponds to a different pair of coordinates, for example: 
            
        ex_XZ[z,x] with z each index of the coordinates np.linspace(z_range/2,-z_range/2,2*int(z_range/z_steps/2)) and x each index for np.linspace(-x_range/2**0.5,x_range/2**0.5,2*int(x_range/x_steps/2**0.5)) in which the field is calculated. 
        
        IMPORTANT: It is worth noting the x positions are sqrt(2) times higher to allow a square field of view for the XY plane (because the maximum radial position to be calculated for the XY plane is higher than the maximum x or y position)
        
        ex_XZ[y,x2] with y each index of the coordinates np.linspace(x_range/2,-x_range/2,2*int(x_range/x_steps/2)) and x each index for np.linspace(-x_range/2,x_range/2,2*int(x_range/x_steps/2)) in which the field is calculated
        
        The XZ plane is given by y=0 and the XY plane by the z parameter 
        
        The unit of the intensity is the same as the initial intensity (mW/cm^2)
    '''
    #moving to wavelength in the optical system's medium
    if multilayer==False:
        alpha=np.arcsin(NA / n)
        wavelength/=n
        f=h*n/NA    
    else:
        alpha=np.arcsin(NA / n[0])
        wavelength/=n[0]
        f=h*n[0]/NA    

    # custom_field_function=lambda theta, phi: np.exp(1j*phi) is the VP mask function
    #Usual parameters for integration precision:
    
    #resolution for field near the focus
    resolution_focus=int(x_range/x_steps)
    resolution_z=int(z_range/z_steps)
    
    #smaller numbers give faster integration times
    
    if callable(custom_field_function)==True:#if a function is given, turn it into the ex_lens and ey_lens variables
        #resolution for field at objective
        if propagation==False:
            #calculate field inciding on the lens by multiplying the phase mask's function and the gaussian amplitude
            ex_lens,ey_lens=generate_incident_field(custom_field_function,alpha,f,divisions_phi,divisions_theta,gamma,beta,w0,I0,wavelength)
            #plot field at the entrance of the lens:
            #since ex_lens and ey_lens are given in theta and phi coordinates, the have to be transformed to cartesian in order to be ploted, hence the name of this function
            if plot_Ei==True:
                print('Showing incident field:')
                plot_in_cartesian(ex_lens,ey_lens,h,alpha,f,figure_name)
        else:
            #calculate field inciding on the lens by fraunhofer's difraction formula
            #N_rho and N_phi are the number of divisions for fraunhoffer's integral by the trapecium's method
            N_rho=200
            N_phi=200                
            ex_lens,ey_lens,I_cartesian,Ex_cartesian,Ey_cartesian=custom_mask_objective_field(h,gamma,beta,divisions_theta,divisions_phi,N_rho,N_phi,alpha,f,custom_field_function,R,L,I0,wavelength,w0,figure_name,plot=True)
            if plot_Ei==True:
                print('Showing incident field:')
                plot_in_cartesian(ex_lens,ey_lens,h,alpha,f,figure_name)

    elif type(custom_field_function)==np.ndarray:#if an array is given
        if propagation==True:
            print('Giving mask function as an array not implemented for calculation of incient field propagation. Can be implemented by obtaining "ex_lens" and "ey_lens" arrays using the "generate_incident_field" function manualy replacing "custom_field_function" with the desired array')
            print('Simulation will continue with propagation=False')
        ex_lens,ey_lens=custom_field_function*np.cos(gamma*np.pi/180)*I0**0.5,custom_field_function*np.sin(gamma*np.pi/180)*np.exp(1j*beta*np.pi/180)*I0**0.5 #make the x and y component based on polarization
        if plot_Ei==True:
            print('Showing incident field:')
            plot_in_cartesian(ex_lens,ey_lens,h,alpha,f,figure_name)
        divisions_phi,divisions_theta=np.shape(custom_field_function)
    else:
        print('Wrong format for mask function, acceptable formats are functions or arrays')
        
    #calculate field at the focal plane:
    if multilayer==False:#no multilayer
        ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,wavelength,z_range,resolution_z,z,resolution_focus,divisions_theta,divisions_phi,x_range)
    else:
        wavelength*=n[0] #to return to wavelength at vacuum
        ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=interface_custom_mask_focus_field_XZ_XY(n,ds,ex_lens,ey_lens,alpha,h,wavelength,z_int,z_range,resolution_z,z,resolution_focus,divisions_theta,divisions_phi,x_range)

    return ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY

