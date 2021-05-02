import numpy as np
def maskfunction(rho,phi,radius,f,k): #parameters radius,f,k are given as an input in case they are used in the phase mask
    '''
    rho, phi are cilindrical coordinates from the field at the entrance of the lens
    
    radius is the beam's width, where the amplitude takes the value of exp(-1)
    
    Because of sine's condition: rho=f*np.sin(theta)

    A gaussian term of the form np.exp(-(rho/radius)**2) is already added by default in the calculus

    #Typical masks:
            
    #VPP:
    phase=np.exp(1j*phi)
    
    
    #Tilted wave front with VPP:     
    gamma=4*10**-5
    phase=np.exp(1j*(phi+k*np.sin(gamma)*rho*np.cos(phi)))
    
    
    #Displaced pattern:  (all units are in mm or radians)
    #Original cylindrical coordinates are (rho,phi), displaced cylindrical coordinates are (rho2,phi2)
    dx=1                #displaced distance on X
    dy=0                #displaced distance on Y
    drho=(dx**2+dy**2)**0.5 
    dphi=np.arctan2(dy,dx)
    rho2=(rho**2+drho**2+2*rho*drho*np.cos(phi-dphi))**0.5
    phi2=np.arctan2(rho*np.sin(phi)+drho*np.sin(dphi),rho*np.cos(phi)+drho*np.cos(dphi))
    phase=np.exp(-(rho2/radius)**2+(rho/radius)**2+1j*phip2) #the +(rho/radius)**2 term is used to cancel the gaussian term that was placed by default on the calculus
    
    #LG10 as the incident amplitude: the raidus of curvatur of the incident beam is approximated as infinite, and the beam's width is consider to be the radius parameter
    c=(2/np.pi)**0.5 #normalization constant
    phase=(rho*2**0.5/radius)*np.exp(-1j*phi) #remember that the exp(-(rho/radius)**2) term is placed by default on the calculus
    
    '''
    
    phase=np.exp(1j*phi)#in case of a new mask, function would be defined here
    
    return phase
    
    