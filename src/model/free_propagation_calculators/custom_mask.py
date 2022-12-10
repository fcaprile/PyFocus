"""
Functions for the simulation of the field obtained by focuisng a gaussian beam
"""
import functools
from scipy import interpolate
from typing import Dict, Tuple

from equations.complex_quadrature import complex_quadrature
from equations.no_phase_mask import load_no_mask_functions
from equations.helpers import cart2pol

import numpy as np
from tqdm import tqdm
from src.equations.gaussian_profile import gaussian_rho

from src.model.free_propagation_calculators.base import FreePropagationCalculator


class CustomMaskFreePropagationCalculator(FreePropagationCalculator):
    
    def calculate(self, gamma=45,beta=-90,steps=500,R=5,L=100,I0=1,wavelength=640,FOV=11,w0=5,limit=2000,div=1,plot=True, figure_name=''):
        raise NotImplementedError
        '''
def custom_mask_objective_field(h,gamma,beta,divisions_theta,divisions_phi,N_rho,N_phi,alpha,focus,custom_field_function,R,L,I0,wavelength,w0,fig_name,plot=True):
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
    '''
    #the function to integrate is:
    f=wavelength rho,phi: rho*custom_field_function(rho,phi)*np.exp(1j*(kl*(rho**2-2*rho*rhop*np.cos(phi-phip))))
    '''
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
    '''