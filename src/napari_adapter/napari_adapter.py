import os
import sys

from ..model.focus_field_calculators.base import FocusFieldCalculator
from ..model.free_propagation_calculators.base import FreePropagationCalculator
from ..custom_dataclasses.field_parameters import FieldParameters, PolarizationParameters
from ..custom_dataclasses.mask import MaskType
from ..custom_dataclasses.interface_parameters import InterfaceParameters
import numpy as np
from ..model.main_calculation_handler import MainCalculationHandler
from scipy.special import binom
from matplotlib import pyplot as plt
class PyFocusSimulator:
    
    def __init__(self,NA, n, wavelength, Nxy, Nz, dr, dz) -> None:
        self.NA = NA
        self.n = n
        self.wavelength = wavelength
        self.Nxy = Nxy
        self.Nz = Nz
        self.dr = dr
        self.dz = dz
        self.incident_field = ...
        
        self._add_zernike_aberration = False
        self._add_cylindrical_lens = False
        self.calculator = MainCalculationHandler(strategy=MaskType.custom_mask)
        
        # Inner variables for later plotting
        self.x = self.y = self.dr * (np.arange(self.Nxy) - self.Nxy // 2)
        self.z = self.dz * (np.arange(self.Nz) - self.Nz // 2)
        self.DeltaX = self.wavelength/self.NA/2 # Abbe resolution
        
        # Inner passage of units
        self._transform_units()
        

    def re_init(self,*args,**kwargs):
        """
        Reinitializes the attributes of the PSFsimulator object wihout creating a new instance (__new__ is not executed)
        """
        self.__init__(*args,**kwargs)
        
    def _transform_units(self):
        ''' Performs a passage from um to nm and calculates the FOV'''
        self.wavelength*=1000*self.n # TODO remover cuando se corrija tema que wavelength se toma dentro o fuera dle medio
        self.dr *= 1000/2**0.5
        self.dz *= 1000/2
        self.radial_FOV = self.dr*self.Nxy*2**0.5
        self.axial_FOV = self.dz*self.Nz*2
        
    
    
    def generate_pupil(self):
        '''Generates the field inciding on the objective lens'''
        pass
    
    def generate_3D_PSF(self):
        basic_parameters = MainCalculationHandler.BasicParameters(file_name='',propagate_incident_field=False,plot_incident_field=False,plot_focus_field_intensity=False,plot_focus_field_amplitude=False)
        polarization = PolarizationParameters(gamma=45, beta=90)
        field_parameters = FieldParameters(w0=50, wavelength=self.wavelength, I_0=1, polarization=polarization)
        objective_field_parameters = FreePropagationCalculator.ObjectiveFieldParameters(L=50, R=10, field_parameters=field_parameters)
        focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=self.NA, n=self.n, h=3, x_steps=self.dr, z_steps=self.dz, x_range=self.radial_FOV, z_range=self.axial_FOV, z=0, phip=0, field_parameters=field_parameters, interface_parameters=None)
        
        mask_function = self._generate_mask_function()
        fields = self.calculator.calculate_3D_fields(basic_parameters=basic_parameters, objective_field_parameters=objective_field_parameters, focus_field_parameters=focus_parameters, mask_function=mask_function)
        fields.calculate_intensity()

        self.PSF3D = fields.Intensity
    
    def _generate_mask_function(self):
        if self._add_zernike_aberration is True:
            return self._nm_polynomial(n=self.N, m=self.M, normalized=False)
        elif self._add_cylindrical_lens is True:
            return 1
        else:
            return 1
    
    def add_slab_scalar(self):
        pass
        
    def write_name(self, basename: str ='') -> str:
        name = '_'.join([basename,
                        f'NA_{self.NA:.1f}',
                        f'n_{self.n:.1f}'])
        
        # if all(hasattr(self, attr) for attr in ["thickness","alpha","n1"]): # slab abberation is there
        #     name = '_'.join([name,
        #                     f'thk_{self.thickness:.0f}',
        #                     f'alpha_{self.alpha:.0f}',
        #                     f'n1_{self.n1:.1f}'])
        
        # if all(hasattr(self, attr) for attr in ["N","M","weight"]): # zernike aberration is there
        #     name = '_'.join([name,
        #                     f'N{self.N}',
        #                     f'M_{self.M}',
        #                     f'w_{self.weight:.1f}'])
        return name
    
    def _nm_polynomial(self, n, m, normalized=True):
        '''Return the function of the zernike polynomial'''
        def nm_normalization(n, m):
            """the norm of the zernike mode n,m in born/wolf convetion
            i.e. sqrt( \int | z_nm |^2 )
            """
            return np.sqrt((1.+(m==0))/(2.*n+2))
        
        if abs(m) > n:
            raise ValueError(" |m| <= n ! ( %s <= %s)" % (m, n))


        def polynomial(rho, phi,*args, **kwargs):
            rho/=2999999
            if (n - m) % 2 == 1:
                return 0
            
            radial = 0
            m0 = abs(m)
            for k in range((n - m0) // 2 + 1):
                radial += (-1.) ** k * binom(n - k, k) * binom(n - 2 * k, (n - m0) // 2 - k) * rho ** (n - 2 * k)

                radial = radial * (rho <= 3.) 

            if normalized:
                prefac = 1. / nm_normalization(n, m) * 2* np.pi* self.weight
            else:
                prefac = 0.5 * 2* np.pi* self.weight
            if m >= 0:
                return np.exp(1j*prefac * radial * np.cos(m0 * phi))
            else:
                return np.exp(1j*prefac * radial * np.sin(m0 * phi))
        return polynomial
    
    def add_Zernike_aberration(self, N, M, weight):
        self._add_zernike_aberration = True
        self.N = N
        self.M = M
        self.weight = weight
    
    def _cylindrical_lens_phase(f_cyl, f):
        def aux(rho, phi,w0,f,k):
            return 2*np.pi * (self.kx)**2 /2/f_cyl * f
        return aux

    def add_cylindrical_lens(self, f_cyl, f):
        """introduce astigmatism placing a thin lens in the pupil
        which produces a quadratic phase in direction x.
        f_cyl is the focal length of the cylindrical lens
        f is the focal length of the objective lens 
        """
        self._add_cylindrical_lens = True
        self.phase += 2*np.pi * (self.kx)**2 /2/f_cyl * f

    def plot_psf_profile(self, dpi = 150):
        '''
        plot the phase a the pupil along x and y
        '''
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=False, dpi=dpi)
        char_size = 12
        sup_title =  f'NA = {self.NA}, n = {self.n}'
        if hasattr(self,'thickness'):
             sup_title += f', slab thickness = {self.thickness} $\mu$m, n1 = {self.n1}, alpha = {self.alpha:.02f}'
        fig.suptitle(sup_title, size=char_size*0.8)
        PSF = self.PSF3D
        Nz, Ny, Nx = PSF.shape
    
        # psf_to_show = PSF.take(indices=Nlist[idx]//2 , axis=idx)
        psf_to_show_x = PSF[Nz//2,Ny//2,:]
        
        psf_to_show_z = PSF[:,Ny//2,Nx//2]
        
        ax[0].plot(self.x, psf_to_show_x,
                   linewidth=1.5)
        ax[0].set_xlabel('x ($\mu$m)',size=char_size)
        ax[0].set_ylabel('PSF', size=char_size)
        ax[0].grid()
        ax[0].plot(np.array([0.,self.DeltaX,1.22*self.DeltaX]),
                   np.array([0.,0.,0.]),
                   'o', markersize=2)
                  
        
        ax[1].plot( self.z, psf_to_show_z,
                   linewidth=1.5)
        ax[1].set_xlabel('z ($\mu$m)', size=char_size)
        # ax[1].set_ylabel('PSF')    
        ax[1].grid()
        DeltaZ = self.wavelength/self.n/(1-np.sqrt(1-self.NA**2/self.n**2)) # Diffraction limited axial resolution
        ax[1].plot(DeltaZ, 0., 'o', markersize=2)
        
        for idx in (0,1):
            ax[idx].xaxis.set_tick_params(labelsize=char_size*0.5)
            ax[idx].yaxis.set_tick_params(labelsize=char_size*0.5)
        
        plt.show()
