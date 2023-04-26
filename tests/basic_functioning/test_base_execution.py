import pytest 
import os
import sys
sys.path.append(os.path.abspath("./src"))

from ...src.model.focus_field_calculators.base import FocusFieldCalculator
from ...src.model.free_propagation_calculators.base import FreePropagationCalculator
from ...src.custom_dataclasses.field_parameters import FieldParameters, PolarizationParameters
from ...src.custom_dataclasses.mask import MaskType
from ...src.custom_dataclasses.interface_parameters import InterfaceParameters
from ...src.model.main_calculation_handler import MainCalculationHandler
import numpy as np

#TODO Numpy broadcast, chat gpt de openAI

def create_base_parameters(base_simulation_parameters=None, field_parameters=None, polarization=None, lens_parameters=None, focus_parameters=None, interface_parameters=None, precise_simulation:bool = False):
    if not base_simulation_parameters:
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
            file_name='test', 
            propagate_incident_field=False,
            plot_incident_field=False, 
            plot_focus_field_amplitude=False,
            plot_focus_field_intensity=False
        )
    
    if not field_parameters:
        if not polarization:
            polarization = PolarizationParameters(gamma=45, beta=90)
        field_parameters = FieldParameters(w0=50, wavelength=640, I_0=1, polarization=polarization)
    
    if not lens_parameters: 
        lens_parameters = FreePropagationCalculator.ObjectiveFieldParameters(L=50, R=10, field_parameters=field_parameters)
    
    if not focus_parameters:
        if precise_simulation: 
            x_steps=10
            z_steps=20
        else:
            x_steps=300
            z_steps=600
        focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=1.4, n=1.5, h=3, x_steps=x_steps, z_steps=z_steps, x_range=1000, z_range=2000, z=0, phip=0, field_parameters=field_parameters, interface_parameters=interface_parameters)    
    
    return base_simulation_parameters, lens_parameters, focus_parameters


def test_no_mask_focus_field():
    do_a_precise_simulation_and_plot_it = False
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_no_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.no_mask)
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters)

def test_with_napari_parameters():
    do_a_precise_simulation_and_plot_it = False
    polarization = PolarizationParameters(gamma=45, beta=90)
    field_parameters = FieldParameters(w0=50, wavelength=532*1.33, I_0=1, polarization=polarization)
    focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=0.65, n=1.33, h=3, x_steps=30, z_steps=200, x_range=1500, z_range=1200, z=0, phip=0, field_parameters=field_parameters, interface_parameters=None)
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_no_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it,
        focus_parameters=focus_parameters
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.custom_mask)
    mask_function = lambda rho, phi,w0,f,k: 1
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=mask_function)

def test_no_mask_focus_field_with_x_polarization():
    do_a_precise_simulation_and_plot_it = False
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_no_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it,
        polarization = PolarizationParameters(gamma=0, beta=0)
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.no_mask)
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters)


def test_VP_mask_focus_field():
    do_a_precise_simulation_and_plot_it = False
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_VP_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.vortex_mask)
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters)
    
def test_custom_mask_focus_field():
    do_a_precise_simulation_and_plot_it = False
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_custom_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.custom_mask)
    mask_function = lambda rho, phi,w0,f,k: np.exp(1j*phi) 
    # mask_function = lambda rho, phi,w0,f,k: 1
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=mask_function)

def test_interface_custom_mask_focus_field():
    do_a_precise_simulation_and_plot_it = False
    interface_parameters = InterfaceParameters(axial_position=0, ns=np.array((1.5,1.5)), ds=np.array((np.inf,np.inf)))
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_interface_custom_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it,
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it,
        interface_parameters=interface_parameters
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.custom_mask)
    
    mask_function = lambda rho, phi,w0,f,k: np.exp(1j*phi) 
    #mask_function = lambda rho, phi,w0,f,k: 1
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=mask_function)

def test_interface_default_mask_focus_field():
    ...

def test_3D_field_custom_mask():
    polarization = PolarizationParameters(gamma=45, beta=90)
    wavelength_0 = 532
    dr = 30
    dz = 200
    Nxy = 51
    Nz = 3
    field_parameters = FieldParameters(w0=50, wavelength=wavelength_0*1.33, I_0=1, polarization=polarization)
    focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=0.65, n=1.33, h=3, x_steps=dr/2**0.5, z_steps=200, x_range=dr*Nxy, z_range=dz*Nz, z=0, phip=0, field_parameters=field_parameters, interface_parameters=None)
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_3D_field_custom_mask', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=False,
                plot_focus_field_intensity=False
            ),
        precise_simulation=False,
        focus_parameters=focus_parameters
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.custom_mask)
    mask_function = lambda rho, phi,w0,f,k: 1
    # mask_function = lambda rho, phi,w0,f,k: np.exp(1j*phi) 
    field = calculation_handler.calculate_3D_fields(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=mask_function)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=False)
    char_size = 12
    sup_title =  f'NA = {focus_parameters.NA}, n = {focus_parameters.n}'
    fig.suptitle(sup_title, size=char_size*0.8)
    field.calculate_intensity()
    PSF = field.Intensity
    Nz, Ny, Nx = PSF.shape
    print(PSF.shape)
    # psf_to_show = PSF.take(indices=Nlist[idx]//2 , axis=idx)
    psf_to_show_x = PSF[Nz//2,Ny//2,:]
    
    psf_to_show_z = PSF[:,Ny//2,Nx//2]
    
    wavelength = wavelength_0
    NA = focus_parameters.NA
    n=focus_parameters.n
    DeltaX = wavelength/NA/2 # Abbe resolution
    x = y = dr * (np.arange(Nxy) - Nxy // 2)
    z = dz * (np.arange(Nz) - Nz // 2)
    ax[0].plot(x, psf_to_show_x,
                linewidth=1.5)
    ax[0].set_xlabel('x ($\mu$m)',size=char_size)
    ax[0].set_ylabel('PSF', size=char_size)
    ax[0].grid()
    ax[0].plot(np.array([0.,DeltaX,1.22*DeltaX]),
                np.array([0.,0.,0.]),
                'o', markersize=2)
                
    
    ax[1].plot( z, psf_to_show_z,
                linewidth=1.5)
    ax[1].set_xlabel('z ($\mu$m)', size=char_size)
    # ax[1].set_ylabel('PSF')    
    ax[1].grid()
    DeltaZ = wavelength/n/(1-np.sqrt(1-NA**2/n**2)) # Diffraction limited axial resolution
    ax[1].plot(DeltaZ, 0., 'o', markersize=2)
    
    for idx in (0,1):
        ax[idx].xaxis.set_tick_params(labelsize=char_size*0.5)
        ax[idx].yaxis.set_tick_params(labelsize=char_size*0.5)
    
    plt.show()

def test_3D_field_custom_mask_with_interface():
    interface_parameters = InterfaceParameters(axial_position=0, ns=np.array((1.5,1.5)), ds=np.array((np.inf,np.inf)))
    polarization = PolarizationParameters(gamma=45, beta=90)
    wavelength_0 = 532
    dr = 30
    dz = 60
    Nxy = 51
    Nz = 11
    field_parameters = FieldParameters(w0=50, wavelength=wavelength_0*1.33, I_0=1, polarization=polarization)
    focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=0.65, n=1.33, h=3, x_steps=dr/2**0.5, z_steps=200, x_range=dr*Nxy, z_range=dz*Nz, z=0, phip=0, field_parameters=field_parameters, interface_parameters=None)
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_3D_field_custom_mask', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=False,
                plot_focus_field_intensity=True
            ),
        precise_simulation=False,
        focus_parameters=focus_parameters,
        interface_parameters=interface_parameters
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.custom_mask)
    mask_function = lambda rho, phi,w0,f,k: 1
    mask_function = lambda rho, phi,w0,f,k: np.exp(1j*phi) 
    field = calculation_handler.calculate_3D_fields(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=mask_function)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=False)
    char_size = 12
    sup_title =  f'NA = {focus_parameters.NA}, n = {focus_parameters.n}'
    fig.suptitle(sup_title, size=char_size*0.8)
    field.calculate_intensity()
    PSF = field.Intensity
    Nz, Ny, Nx = PSF.shape
    print(PSF.shape)
    # psf_to_show = PSF.take(indices=Nlist[idx]//2 , axis=idx)
    psf_to_show_x = PSF[Nz//2,Ny//2,:]
    
    psf_to_show_z = PSF[:,Ny//2,Nx//2]
    
    wavelength = wavelength_0
    NA = focus_parameters.NA
    n=focus_parameters.n
    DeltaX = wavelength/NA/2 # Abbe resolution
    x = y = dr * (np.arange(Nxy) - Nxy // 2)
    z = dz * (np.arange(Nz) - Nz // 2)
    ax[0].plot(x, psf_to_show_x,
                linewidth=1.5)
    ax[0].set_xlabel('x ($\mu$m)',size=char_size)
    ax[0].set_ylabel('PSF', size=char_size)
    ax[0].grid()
    ax[0].plot(np.array([0.,DeltaX,1.22*DeltaX]),
                np.array([0.,0.,0.]),
                'o', markersize=2)
                
    
    ax[1].plot( z, psf_to_show_z,
                linewidth=1.5)
    ax[1].set_xlabel('z ($\mu$m)', size=char_size)
    # ax[1].set_ylabel('PSF')    
    ax[1].grid()
    DeltaZ = wavelength/n/(1-np.sqrt(1-NA**2/n**2)) # Diffraction limited axial resolution
    ax[1].plot(DeltaZ, 0., 'o', markersize=2)
    
    for idx in (0,1):
        ax[idx].xaxis.set_tick_params(labelsize=char_size*0.5)
        ax[idx].yaxis.set_tick_params(labelsize=char_size*0.5)
    
    plt.show()

