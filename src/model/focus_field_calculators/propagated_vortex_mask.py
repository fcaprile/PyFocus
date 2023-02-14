
from model.focus_field_calculators.base import FocusFieldCalculator
from equations.helpers import cart2pol

import numpy as np
from tqdm import tqdm
# import os, sys
# sys.path.append(os.getcwd())

# from src.equations.radial_simmetry import load_radial_simmetry_mask_functions


class PropagatedVortexMaskFocusFieldCalculator(FocusFieldCalculator):
    MATRIX_AMOUNT: int = 5
    DESCRIPTION: str = 'Rotation simmetry mask calulation'
    
    def calculate(self, E_rho,alpha,beta,gamma,n,f,w0,wavelength,I0,x_range,z_range,z_steps,x_steps, zp0, phip0):        
        functions_to_integrate = load_radial_simmetry_mask_functions(f,w0)
        
        E_theta=lambda theta: E_rho(np.sin(theta)*f)
        functions_to_integrate = self.INTEGRATION_EQUATIONS(f,w0, E_theta)
        
        matrixes = self._integrate(self.MATRIX_AMOUNT, functions_to_integrate, wavelength, alpha, z_range, x_range, z_steps, r_steps, self.description)
        matrixes = self._mirror_on_z_axis(matrixes)
        field = self._calculate_2D_fields(matrixes,wavelength,I0,beta,gamma,z_steps, r_steps, x_steps,x_range,z_range,phip0,n,f,zp0)
                
        return field


    def _calculate_2D_fields(self, input_matrixes,wavelength,I0,beta,gamma,z_steps, r_steps, x_steps,x_range,z_range,phip0,n,f,zp0):
        a1, a2 = self._calculate_amplitude_factors(I0, gamma, beta, wavelength, f)
        I1, I2, I3, I4, I5 = input_matrixes            

        x_size, y_size = self._calculate_matrix_size(x_range=x_range, x_steps=x_steps)
        exx, eyx, ezx, exy, eyy, ezy = self._initialize_fields(x_size=x_size, y_size=y_size)

        for xx in range(x_size):
            for yy in range(y_size):
                xcord=xx - np.rint(2*r_steps /np.sqrt(2))/2#not sure of multiplIng by 2 and dividing by 2 outside the int, i thought it was to be sure to get the 0,0 at xx=np.rint(2*r_steps /np.sqrt(2))/2
                ycord=yy - np.rint(2*r_steps /np.sqrt(2))/2
                phip,rp=cart2pol(xcord+1,ycord+1)#nuevamente el +1 es para no tener problemas
                rp=int(np.rint(rp))
                exx[yy,xx]=a1*(I1[rp]*np.exp(1j*phip) - 0.5*I2[rp]*np.exp(-1j*phip) + 0.5*I3[rp]*np.exp(3j*phip))
                eyx[yy,xx]=- 0.5*a1*1j*(I2[rp]*np.exp(- 1j*phip) + I3[rp]*np.exp(3j*phip))
                ezx[yy,xx]=a1*1j*(I4[rp] - I5[rp]*np.exp(2j*phip))
                exy[yy,xx]=-0.5*a2*1j*(I2[rp]*np.exp(- 1j*phip) +I3[rp]*np.exp(3j*phip))
                eyy[yy,xx]=a2*(I1[rp]*np.exp(1j*phip) + 0.5*I2[rp]*np.exp(- 1j*phip) - 0.5*I3[rp]*np.exp(3j*phip))
                ezy[yy,xx]=- a2*(I4[rp] + I5[rp]*np.exp(2j*phip))
        Ex=exx + exy
        Ey=eyx + eyy
        Ez=ezx + ezy

        return self.FieldAtFocus(None, None, None, Ex,Ey,Ez)
