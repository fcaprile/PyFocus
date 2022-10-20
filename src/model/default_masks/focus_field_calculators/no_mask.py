"""
Functions for the simulation of the field obtained by focuisng a gaussian beam
"""
import functools
from typing import Dict, List, Tuple

from model.default_masks.focus_field_calculators.base import DefaultMaskFocusFieldCalculator
from equations.complex_quadrature import complex_quadrature
from equations.no_phase_mask import load_no_mask_functions
from equations.helpers import cart2pol

import numpy as np
from tqdm import tqdm


class NoMaskFocusFieldCalculator(DefaultMaskFocusFieldCalculator):
    MATRIX_AMOUNT: int = 3
    INTEGRATION_EQUATIONS: callable = load_no_mask_functions
    DESCRIPTION: str = 'No mask calulation'
    
    def execute(self,alpha,beta,gamma,n,f,w0,wavelength,I0,x_range,z_range,z_steps,x_steps, zp0, phip0):        
        ztotalsteps, rtotalsteps = self.calculate_steps(z_range, z_steps, x_range, x_steps)    
        functions_to_integrate = self.INTEGRATION_EQUATIONS(f,w0)
        
        matrixes = self.integrate(self.MATRIX_AMOUNT, functions_to_integrate, wavelength, alpha, z_range, x_range, ztotalsteps, rtotalsteps, self.description)
        matrixes = self.mirror_on_z_axis(matrixes)
        field = self._calculate_field(matrixes,wavelength,I0,beta,gamma,ztotalsteps, rtotalsteps, x_steps,x_range,z_range,phip0,n,f,zp0)
        
        return field
        
    def _calculate_field(self, input_matrixes,wavelength,I0,beta,gamma,ztotalsteps, rtotalsteps, x_steps,x_range,z_range,phip0,n,f,zp0):
        a1, a2 = self.calculate_factors(I0, gamma, beta, wavelength, f)
        II1, II2, II3 = input_matrixes
        
        ######################xz plane#######################
        #for negative z values there is a minus sign that comes out, and so the first part of the vstack has a - multiplyed
        exx=-a1*1j*np.hstack((np.fliplr(II1)+np.cos(2*phip)*np.fliplr(II3), II1[:,1:rtotalsteps-1]+np.cos(2*phip)*II3[:,1:rtotalsteps-1]))
        eyx=-a1*1j*np.hstack((np.fliplr(II3)*np.sin(2*phip), np.sin(2*phip)*II3[:,1:rtotalsteps-1]))
        ezx=a1*2*np.hstack((-np.fliplr(II2)*np.cos(phip), np.cos(phip)*II2[:,1:rtotalsteps-1]))
        
        exy=-a2*1j*np.hstack((np.fliplr(II3)*np.sin(2*phip), np.sin(2*phip)*II3[:,1:rtotalsteps-1]))
        eyy=-a2*1j*np.hstack((np.fliplr(II1)-np.cos(2*phip)*np.fliplr(II3), II1[:,1:rtotalsteps-1]-np.cos(2*phip)*II3[:,1:rtotalsteps-1]))
        ezy=-a2*2*np.hstack((-np.fliplr(II2)*np.sin(phip), np.sin(phip)*II2[:,1:rtotalsteps-1]))
        
        Ex=exx+exy
        Ey=eyx+eyy
        Ez=ezx+ezy
        
        ######################xy plane#######################
        #index 2 represents it's calculated on the xy plane
        x_size, y_size = self._calculate_matrix_size(x_range=x_range, x_steps=x_steps)
        exx2, eyx2, ezx2, exy2, eyy2, ezy2 = self._initialize_fields(x_size=x_size, y_size=y_size)
        zz=ztotalsteps + int(np.rint(zp0/z_range*2*ztotalsteps))  #zz signals to the row of kz=kz0 in each II
        for xx in range(x_size):
            for yy in range(y_size):
                xcord = xx - int(np.rint(x_range/2/x_steps))+1
                ycord = -yy + int(np.rint(x_range/2/x_steps))-1
                phip, rp = cart2pol(xcord,ycord, round_r=True)
                
                exx2[yy,xx]=-a1*1j*(II1[zz,rp]+np.cos(2*phip)*II3[zz,rp])
                eyx2[yy,xx]=-a1*1j*(np.sin(2*phip)*II3[zz,rp])
                ezx2[yy,xx]=a1*2*(np.cos(phip)*II2[zz,rp])
                exy2[yy,xx]=-a2*1j*(np.sin(2*phip)*II3[zz,rp])
                eyy2[yy,xx]=-a2*1j*(II1[zz,rp]-np.cos(2*phip)*II3[zz,rp])
                ezy2[yy,xx]=-a2*2*(np.sin(phip)*II2[zz,rp])
        
        Ex2=exx2+exy2
        Ey2=eyx2+eyy2
        Ez2=ezx2+ezy2
        
        return Ex,Ey,Ez,Ex2,Ey2,Ez2

