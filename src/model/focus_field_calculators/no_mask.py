"""
Functions for the simulation of the field obtained by focuisng a gaussian beam
"""
from typing import Dict, Tuple

from model.focus_field_calculators.base import FocusFieldCalculator
from equations.complex_quadrature import complex_quadrature
from equations.no_phase_mask import load_no_mask_functions
from equations.helpers import cart2pol

import numpy as np
from tqdm import tqdm


class NoMaskFocusFieldCalculator(FocusFieldCalculator):
    MATRIX_AMOUNT: int = 3
    DESCRIPTION: str = 'No mask calulation'
    
    def calculate(self, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters):
        functions_to_integrate = load_no_mask_functions(focus_field_parameters.f,focus_field_parameters.field_parameters.w0)
        
        matrixes = self._integrate(self.MATRIX_AMOUNT, functions_to_integrate, focus_field_parameters, self.DESCRIPTION)
        matrixes = self._mirror_on_z_axis(matrixes)
        field = self._calculate_field(matrixes, focus_field_parameters)
        
        return field
        
    def _calculate_field(self, input_matrixes,focus_field_parameters: FocusFieldCalculator.FocusFieldParameters):
        a1, a2 = self._calculate_amplitude_factors(focus_field_parameters)
        phip = focus_field_parameters.phip
        II1, II2, II3 = input_matrixes
        
        ######################xz plane#######################
        #for negative z values there is a minus sign that comes out, and so the first part of the vstack has a - multiplyed
        exx=-a1*1j*np.hstack((np.fliplr(II1)+np.cos(2*phip)*np.fliplr(II3), II1[:,1:focus_field_parameters.r_steps-1]+np.cos(2*phip)*II3[:,1:focus_field_parameters.r_steps-1]))
        eyx=-a1*1j*np.hstack((np.fliplr(II3)*np.sin(2*phip), np.sin(2*phip)*II3[:,1:focus_field_parameters.r_steps-1]))
        ezx=a1*2*np.hstack((-np.fliplr(II2)*np.cos(phip), np.cos(phip)*II2[:,1:focus_field_parameters.r_steps-1]))
        
        exy=-a2*1j*np.hstack((np.fliplr(II3)*np.sin(2*phip), np.sin(2*phip)*II3[:,1:focus_field_parameters.r_steps-1]))
        eyy=-a2*1j*np.hstack((np.fliplr(II1)-np.cos(2*phip)*np.fliplr(II3), II1[:,1:focus_field_parameters.r_steps-1]-np.cos(2*phip)*II3[:,1:focus_field_parameters.r_steps-1]))
        ezy=-a2*2*np.hstack((-np.fliplr(II2)*np.sin(phip), np.sin(phip)*II2[:,1:focus_field_parameters.r_steps-1]))
        
        Ex=exx+exy
        Ey=eyx+eyy
        Ez=ezx+ezy
        
        ######################xy plane#######################
        #index 2 represents it's calculated on the xy plane
        x_size, y_size = self._calculate_matrix_size(x_range=focus_field_parameters.x_range, x_steps=focus_field_parameters.x_steps)
        exx2, eyx2, ezx2, exy2, eyy2, ezy2 = self._initialize_fields(x_size=x_size, y_size=y_size)
        zz=focus_field_parameters.z_step_count + int(np.rint(focus_field_parameters.z/focus_field_parameters.z_range*2*focus_field_parameters.z_step_count)) -1  #zz signals to the row of kz=kz0 in each II
        
        for xx in range(x_size):
            for yy in range(y_size):
                xcord = xx - int(np.rint(focus_field_parameters.x_range/2/focus_field_parameters.x_steps))+1
                ycord = -yy + int(np.rint(focus_field_parameters.x_range/2/focus_field_parameters.x_steps))-1
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
        
        return self.FieldAtFocus(Ex,Ey,Ez,Ex2,Ey2,Ez2)

