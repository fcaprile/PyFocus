"""
Functions for the simulation of the field obtained by focuisng a gaussian beam
"""
import numpy as np
from complex_quadrature import complex_quadrature
from tqdm import tqdm
from scipy.special import jv

def no_mask_integration(alpha,n,f,w0,wavelength,x_range,z_range,z_steps,r_steps):
    '''
    Generate the II arrays, which are the result of the integration for different positions along the radius and z
    
    This matrixes are later used to calculate the focused field
    
    Args:        
        :alpha: semiangle of aperture
                
        :wavelength: wavelength in the medium (equals wavelength in vacuum/n)
        
        :r_steps: resolution in the x or y coordinate (nm)
        
        :z_steps: resolution in the axial coordinate,z (nm)
        
        :x_range: field of view in the x or y coordinate in which the field is calculated (nm)
        
        :z_range: field of view in the axial coordinate, z, in which the field is calculated (nm)

        The other parameters are specified in sim
        
    Returns:
        :(arrays): II1,II2,II3
    '''
    #passage to nm:
    f*=10**6
    w0*=10**6

    ztotalsteps=int(np.rint(z_range/z_steps/2))
    rtotalsteps=int(np.rint(x_range/r_steps/2**0.5)) #the actual field of view of the X axis in the XZ plane will be x_range*2**0.5

    gaussian=lambda theta:np.exp(-(np.sin(theta)*f/w0)**2)#incident gaussian beam's amplitude

    I1=np.zeros((ztotalsteps,rtotalsteps),dtype=complex)
    I2=np.copy(I1)
    I3=np.copy(I1)

    fun1=lambda theta: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1+np.cos(theta))*jv(0,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun2=lambda theta: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)**2*jv(1,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun3=lambda theta: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1-np.cos(theta))*jv(2,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))    

    for zz in tqdm(range(ztotalsteps),desc='No mask calulation'):
        for rr in range(rtotalsteps):
            kz=zz*2*np.pi/wavelength/ztotalsteps*z_range/2 
            kr=rr*2*np.pi/wavelength/rtotalsteps*x_range/2*2**0.5
            I1[zz,rr]= complex_quadrature(fun1,0,alpha)[0]
            I2[zz,rr]= complex_quadrature(fun2,0,alpha)[0]
            I3[zz,rr]= complex_quadrature(fun3,0,alpha)[0]


    II1=np.vstack((np.flipud(np.conj(I1)),I1[1:,:]))
    II2=np.vstack((np.flipud(np.conj(I2)),I2[1:,:]))
    II3=np.vstack((np.flipud(np.conj(I3)),I3[1:,:]))


    return II1,II2,II3


def no_mask_fields(II1,II2,II3,wavelength,I0,beta,gamma,z_steps,r_steps,x_range,z_range,phip0,n,f,zp0):
    '''
    Given the II matrixes calculate the field on the focus
    
    Args: 
        :phip0: Azimutal offset for the XZ plane calculus
    
        :wavelength: wavelength given in the medium (equals wavelength in vacuum/n)
        
        :zp0: axial position of the XY plane

        The other parameters are specified in sim
    
    Returns:        
        :arrays: Ex,Ey,Ez,Ex2,Ey2,Ez2, each one is a matrix with the amplitude of each cartesian component on the XZ plane (Ex,Ey,Ez) or on the XY plane (Ex2,Ey2,Ez2)
    
        Each index of the matrixes corresponds to a different pair of coordinates, for example: 
            
        ex[z,x] with z each index of the coordinates np.linspace(z_range/2,-z_range/2,2*int(z_range/z_steps/2)) and x each index for np.linspace(-x_range/2**0.5,x_range/2**0.5,2*int(x_range/r_steps/2**0.5)) in which the field is calculated
        
        ex2[y,x2] with y each index of the coordinates np.linspace(x_range/2,-x_range/2,2*int(x_range/r_steps/2)) and x each index for np.linspace(-x_range/2,x_range/2,2*int(x_range/r_steps/2)) in which the field is calculated
        
        The XZ plane is given by y=0 and the XZ plane by z=zp0 
        
        The radial field of view in the XZ plane is sqrt(2) times bigger to allow a square field of view for the XY plane (the maximum radial position is higher than the maximum x or y position)
    '''
    #passage to nm:
    f*=10**6

    ztotalsteps=int(np.rint(z_range/z_steps/2))
    rtotalsteps=int(np.rint(x_range/r_steps/2**0.5))

    def cart2pol(x,y):    
        r = np.sqrt(x**2+y**2)
        t = np.arctan2(y,x)
        return t,r
    #transform to radians:
    beta*= np.pi/180
    phip=phip0 / 180*np.pi
    gamma*= np.pi/180
    
    #E1,E2 are the components of the polarization
    E1=np.sqrt(I0)*np.cos(gamma)/wavelength*np.pi*f
    E2=np.sqrt(I0)*np.sin(gamma)/wavelength*np.pi*f
    a1=np.copy(E1)
    a2=E2*np.exp(1j*beta)

    ######################xz plane#######################
    #for negative z values there is a minus sign that comes out, and so the first part of the vstack has a - multiplyed
    exx=-a1*1j*np.hstack((np.fliplr(II1)+np.cos(2*phip)*np.fliplr(II3), II1[:,1:rtotalsteps-1]+np.cos(2*phip)*II3[:,1:rtotalsteps-1]))
    eyx=-a1*1j*np.hstack((np.fliplr(II3)*np.sin(2*phip), np.sin(2*phip)*II3[:,1:rtotalsteps-1]))
    ezx=-a1*2*np.hstack((-np.fliplr(II2)*np.cos(phip), np.cos(phip)*II2[:,1:rtotalsteps-1]))

    exy=-a2*1j*np.hstack((np.fliplr(II3)*np.sin(2*phip), np.sin(2*phip)*II3[:,1:rtotalsteps-1]))
    eyy=-a2*1j*np.hstack((np.fliplr(II1)-np.cos(2*phip)*np.fliplr(II3), II1[:,1:rtotalsteps-1]-np.cos(2*phip)*II3[:,1:rtotalsteps-1]))
    ezy=-a2*2*np.hstack((-np.fliplr(II2)*np.sin(phip), np.sin(phip)*II2[:,1:rtotalsteps-1]))

    Ex=exx+exy
    Ey=eyx+eyy
    Ez=ezx+ezy

    ######################xy plane#######################
    #index 2 represents it's calculated on the xy plane

    x2,y2=(int(np.rint(x_range/r_steps/2-1)*2),int(np.rint(x_range/r_steps/2-1)*2))
    exx2=np.zeros((x2,y2),dtype=complex)
    eyx2=np.zeros((x2,y2),dtype=complex)
    ezx2=np.zeros((x2,y2),dtype=complex)
    exy2=np.zeros((x2,y2),dtype=complex)
    eyy2=np.zeros((x2,y2),dtype=complex)
    ezy2=np.zeros((x2,y2),dtype=complex)
    zz=ztotalsteps + int(np.rint(zp0/z_range*2*ztotalsteps))  #zz signals to the row of kz=kz0 in each II
    for xx in range(x2):
        for yy in range(y2):
            xcord=xx - int(np.rint(x_range/2/r_steps))+1
            ycord=-yy + int(np.rint(x_range/2/r_steps))-1
            phip,rp=cart2pol(xcord,ycord)#nuevamente el +1 es para no tener problemas
            rp=int(np.rint(rp))
            exx2[yy,xx]=-a1*1j*(II1[zz,rp]+np.cos(2*phip)*II3[zz,rp])
            eyx2[yy,xx]=-a1*1j*(np.sin(2*phip)*II3[zz,rp])
            ezx2[yy,xx]=-a1*2*(np.cos(phip)*II2[zz,rp])
            exy2[yy,xx]=-a2*1j*(np.sin(2*phip)*II3[zz,rp])
            eyy2[yy,xx]=-a2*1j*(II1[zz,rp]-np.cos(2*phip)*II3[zz,rp])
            ezy2[yy,xx]=-a2*2*(np.sin(phip)*II2[zz,rp])
    Ex2=exx2+exy2
    Ey2=eyx2+eyy2
    Ez2=ezx2+ezy2
    
    
    return Ex,Ey,Ez,Ex2,Ey2,Ez2
    


