"""
Functions for the simulation of the foci obtained by a VP mask
"""
from equations.vortex_phase_mask import load_vortex_mask_functions
from equations.helpers import cart2pol
from model.focus_field_calculators.base import FocusFieldCalculator

import numpy as np


class VortexMaskFocusFieldCalculator(FocusFieldCalculator):
    MATRIX_AMOUNT: int = 5
    DESCRIPTION: str = 'Vortex mask calulation'
    
    def calculate(self, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters):
        functions_to_integrate = load_vortex_mask_functions(focus_field_parameters.f,focus_field_parameters.field_parameters.w0)
        
        matrixes = self.integrate(self.MATRIX_AMOUNT, functions_to_integrate, focus_field_parameters, self.DESCRIPTION)
        matrixes = self.mirror_on_z_axis(matrixes)
        field = self._calculate_field(matrixes, focus_field_parameters)
        
        return field
        
    def _calculate_field(self, input_matrixes,wavelength,I0,beta,gamma,ztotalsteps, rtotalsteps, x_steps,x_range,z_range,phip0,n,f,zp0):
        a1, a2 = self._calculate_amplitude_factors(I0, gamma, beta, wavelength, f)
        II1, II2, II3, II4, II5 = input_matrixes

        ######################xz plane#######################
        #for negative z values there is a minus sign that comes out, and so the first part of the vstack has a - multiplyed
        exx=a1*np.hstack((- np.fliplr(II1)*np.exp(1j*phip) + 0.5*np.fliplr(II2)*np.exp(- 1j*phip) - 0.5*np.fliplr(II3)*np.exp(3j*phip),II1[:,1:rtotalsteps-1]*np.exp(1j*phip) - 0.5*II2[:,1:rtotalsteps-1]*np.exp(- 1j*phip) + 0.5*II3[:,1:rtotalsteps-1]*np.exp(3j*phip)))
        eyx=-0.5*1j*a1*np.hstack((- np.fliplr(II2)*np.exp(- 1j*phip) - np.fliplr(II3)*np.exp(3j*phip),II2[:,1:rtotalsteps-1]*np.exp(- 1j*phip) + II3[:,1:rtotalsteps-1]*np.exp(3j*phip)))
        ezx=-a1*1j*np.hstack((np.fliplr(II4) - np.fliplr(II5)*np.exp(2j*phip),II4[:,1:rtotalsteps-1] - II5[:,1:rtotalsteps-1]*np.exp(2j*phip)))
        
        exy=- 0.5*a2*1j*np.hstack((- np.fliplr(II2)*np.exp(- 1j*phip) - np.fliplr(II3)*np.exp(3j*phip),II2[:,1:rtotalsteps-1]*np.exp(- 1j*phip) + II3[:,1:rtotalsteps-1]*np.exp(3j*phip)))
        eyy=a2*np.hstack((- np.fliplr(II1)*np.exp(1j*phip) - 0.5*np.fliplr(II2)*np.exp(- 1j*phip) + 0.5*np.fliplr(II3)*np.exp(3j*phip),II1[:,1:rtotalsteps-1]*np.exp(1j*phip) + 0.5*II2[:,1:rtotalsteps-1]*np.exp(- 1j*phip) - 0.5*II3[:,1:rtotalsteps-1]*np.exp(3j*phip)))
        ezy=a2*np.hstack((np.fliplr(II4) + np.fliplr(II5)*np.exp(2j*phip),II4[:,1:rtotalsteps-1] +II5[:,1:rtotalsteps-1]*np.exp(2j*phip)))

        Ex=exx + exy
        Ey=eyx + eyy
        Ez=ezx + ezy

        ######################xy plane#######################
        #index 2 represents it's calculated on the xy plane

        x_size, y_size = self._calculate_matrix_size(x_range=x_range, x_steps=x_steps)
        exx2, eyx2, ezx2, exy2, eyy2, ezy2 = self._initialize_fields(x_size=x_size, y_size=y_size)
        zz=ztotalsteps + int(np.rint(zp0/z_range*2*ztotalsteps))  #zz signals to the row of kz=kz0 in each II
        for xx in range(x_size):
            for yy in range(y_size):
                xcord=xx - int(np.rint(x_range/2/x_steps))+1
                ycord=-yy + int(np.rint(x_range/2/x_steps))-1
                phip,rp=cart2pol(xcord,ycord)#nuevamente el +1 es para no tener problemas
                rp=int(np.rint(rp))
                exx2[yy,xx]=a1*(II1[zz,rp]*np.exp(1j*phip) - 0.5*II2[zz,rp]*np.exp(-1j*phip) + 0.5*II3[zz,rp]*np.exp(3j*phip))
                eyx2[yy,xx]=- 0.5*a1*1j*(II2[zz,rp]*np.exp(- 1j*phip) + II3[zz,rp]*np.exp(3j*phip))
                ezx2[yy,xx]=-a1*1j*(II4[zz,rp] - II5[zz,rp]*np.exp(2j*phip))
                exy2[yy,xx]=-0.5*a2*1j*(II2[zz,rp]*np.exp(- 1j*phip) +II3[zz,rp]*np.exp(3j*phip))
                eyy2[yy,xx]=a2*(II1[zz,rp]*np.exp(1j*phip) + 0.5*II2[zz,rp]*np.exp(- 1j*phip) - 0.5*II3[zz,rp]*np.exp(3j*phip))
                ezy2[yy,xx]=a2*(II4[zz,rp] + II5[zz,rp]*np.exp(2j*phip))
        Ex2=exx2 + exy2
        Ey2=eyx2 + eyy2
        Ez2=ezx2 + ezy2

        return Ex,Ey,Ez,Ex2,Ey2,Ez2


