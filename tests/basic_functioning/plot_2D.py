import matplotlib.pyplot as plt
import numpy as np

def plot_along_z_and_x(field, focus_parameters, wavelength_0, dr, dz, Nxy):
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
    
    # plt.show()
