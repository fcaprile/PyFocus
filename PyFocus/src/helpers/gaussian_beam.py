def generate_gaussian_objective_field(objective_field_parameters):
#def generate_rotated_incident_field(maskfunction,alpha,f,divisions_phi,divisions_theta,gamma,beta,w0,I0,wavelength):
    '''
    Generate a matrix for the field X and Y direction of the incident field on the lens evaluated at phi-180º for ex_lens and at phi-270º for ey_lens, given the respective maskfunction
    
    Args:        
        :maskfunction (function): Analytical function that defines the phase mask, must be a function of the 5 internal variables: rho, phi, w0, f and k, with:
            
            rho: Radial coordinate from 0 to the aperture radius of the objective.
            
            phi: Azimutal coordinate from 0 to 2pi.
            
            w0: Radius of the incident gaussian beam.
            
            f: Focal distane of the objective lens (mm)
            
            k: Wavenumber in the objective lens medium (mm)
            
            The real part defines the amplitude of the incident field
    
        :divisions_phi,divisions_theta: Number of divisions in the phi and theta coordinates to use the 2D integration for the calculation of the focused field
 
        The rest of the parameters are specified in sim

    Returns:
        :arrays: ex_lens,ey_lens
        
    This arrays have the value of the amplitude of the incident field for each value of theta and phi: ex_lens[phi_position,theta_position]
    for phi_position a value in np.linspace(0,2*np.pi,divisions_phi) and theta_position a value in np.linspace(0,alpha,divisions_theta) 
    '''
    wavelength/=10**6#pasage from nm to mm
    k=2*np.pi/wavelength #k is given in nm, the same as wavelength
    ex_lens=np.zeros((divisions_phi,divisions_theta),dtype=complex)
    ey_lens=np.zeros((divisions_phi,divisions_theta),dtype=complex)

    theta_values=np.linspace(0,alpha,divisions_theta)  #divisions of theta in which the trapezoidal 2D integration is done
    rho_values=np.sin(theta_values)*f              #given by the sine's law
    phi_values_x=np.linspace(np.pi,3*np.pi,divisions_phi)   #divisions of phi in which the trapezoidal 2D integration is done
    phi_values_y=np.linspace(3*np.pi/2,7*np.pi/2,divisions_phi)   #divisions of phi in which the trapezoidal 2D integration is done
    for i,phi in enumerate(phi_values_x):
        for j,rho in enumerate(rho_values):
            phase=maskfunction(rho,phi,w0,f,k)
            ex_lens[i,j]=phase
    for i,phi in enumerate(phi_values_y):
        for j,rho in enumerate(rho_values):
            phase=maskfunction(rho,phi,w0,f,k)
            ey_lens[i,j]=phase
    ex_lens*=np.cos(gamma*np.pi/180)*I0**0.5
    ey_lens*=np.sin(gamma*np.pi/180)*np.exp(1j*beta*np.pi/180)*I0**0.5
    
    return ex_lens,ey_lens
