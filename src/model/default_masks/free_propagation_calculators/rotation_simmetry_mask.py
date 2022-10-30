
import functools
from typing import Dict, List, Tuple

from equations.complex_quadrature import complex_quadrature
from equations.helpers import cart2pol

import numpy as np
from tqdm import tqdm

from src.model.default_masks.free_propagation_calculators.base import DefaultMaskFreePropagationCalculator


class RotationSimmetryFreePropagationCalculator(DefaultMaskFreePropagationCalculator):
    
    def execute(self, gamma=45,beta=-90,steps=500,R=5,L=100,I0=1,wavelength=640,FOV=11,w0=5,limit=2000,div=1,plot=True, figure_name=''):
        raise NotImplementedError
        '''
        # calculating the rho values in wich to integrate
        rmax=FOV
        k=2*np.pi/wavelength 
        rvalues=np.linspace(0,rmax*2**0.5,steps)
        
        # Incident field is a gaussean beam
        E_xy = gaussian_rho(w0) 
        a1, a2 = self.calculate_factors(I0, gamma, beta, wavelength)    
        fun=lambda rho: E_xy(rho)*rho*np.exp(1j*np.pi/wavelength/L*rho**2)*jv(1,k/L*rho*rhop) # TODO actualizar la función a la de un haz gaussiano que se propaga
        
        Int=np.zeros(steps,dtype=complex)
        
        for i in tqdm(range(steps),desc='Calculating field at the objective'):
            rhop=rvalues[i]
            for l in range(div):
                Int[i]+=complex_quadrature(fun,R*l/div,R*(l+1)/div,lim=limit)[0]
        
        #interpolating the Integration for values of rho:
        Int_interpolated=interpolate.interp1d(rvalues,Int,kind='cubic')    
        
        #calculating the field along the X, Y plane
        xmax=FOV/2
        xyvalues=np.linspace(-xmax,xmax,int(np.rint(steps/2**0.5/4)))
        tot_xy=len(xyvalues)
        
        Ex, Ey, Ifield = [np.zeros((tot_xy,tot_xy),dtype=complex) for _ in range(3)]
        
        E_fun=lambda rho:2*np.pi*np.exp(1j*k*(L+rho**2/2/L))/wavelength/L*Int_interpolated(rho)
        
        for i,xp in enumerate(xyvalues): 
            for j,yp in enumerate(xyvalues): 
                rhop=(xp**2+yp**2)**0.5
                phip=np.arctan2(yp,xp)
                Ex[j,i]=a1*E_fun(rhop)*np.exp(1j*phip)
                Ey[j,i]=a2*E_fun(rhop)*np.exp(1j*phip)
        
        x ,y = tot_xy, tot_xy
        Ifield=np.zeros((x,y))
        for i in range(x):
            for j in range(y):
                Ifield[j,i]=np.real(Ex[j,i]*np.conj(Ex[j,i])+Ey[j,i]*np.conj(Ey[j,i]))
        
        if plot==True:
            self.plot_function(Ifield, Ex, Ey, xmax=xmax, figure_name=figure_name)
        
        return E_fun,Ex,Ey
        '''