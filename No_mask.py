# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:34:55 2020

@author: ferchi
"""
import numpy as np
from complex_quadrature import complex_quadrature
from tqdm import tqdm
from scipy.special import jv

def no_mask_integration(alpha,n,f,w0,wavelength,field_of_view,z_field_of_view,zsteps,rsteps):
    '''
    Generate the II matrixes, which are the result of the integration for different positions along the radius and z
    This matrixes are later used to calculate the field
    wavelength is given in the medium (equals wavelength in vacuum/n)
    '''

    ztotalsteps=np.int(np.rint(z_field_of_view/zsteps/2))
    rtotalsteps=np.int(np.rint(field_of_view/rsteps*2**0.5/2)) #the actual field of view of the X axis in the XZ plane will be field_of_view*2**0.5

    gaussian=lambda theta:np.exp(-(np.sin(theta)*f/w0)**2)

    I1=np.zeros((ztotalsteps,rtotalsteps),dtype=complex)
    I2=np.copy(I1)
    I3=np.copy(I1)

    fun1=lambda theta: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1+np.cos(theta))*jv(0,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun2=lambda theta: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)**2*jv(1,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun3=lambda theta: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1-np.cos(theta))*jv(2,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))    

    for zz in tqdm(range(ztotalsteps),desc='No mask calulation'):
        for rr in range(rtotalsteps):
            kz=zz*2*np.pi/wavelength/ztotalsteps*z_field_of_view/2 
            kr=rr*2*np.pi/wavelength/rtotalsteps*field_of_view/2*2**0.5
            I1[zz,rr]= complex_quadrature(fun1,0,alpha)[0]
            I2[zz,rr]= complex_quadrature(fun2,0,alpha)[0]
            I3[zz,rr]= complex_quadrature(fun3,0,alpha)[0]


    II1=np.vstack((np.flipud(np.conj(I1)),I1[1:,:]))
    II2=np.vstack((np.flipud(np.conj(I2)),I2[1:,:]))
    II3=np.vstack((np.flipud(np.conj(I3)),I3[1:,:]))


    return II1,II2,II3


def no_mask_fields(II1,II2,II3,wavelength,I0,beta,gamma,zsteps,rsteps,field_of_view,z_field_of_view,phip0,n,f,zp0):
    '''
    Given the II matrixes calculate the field on the focus
    parameter phip0 gives an azimutal offset for the XZ plane calculus
    wavelength is given in the medium (equals wavelength in vacuum/n)
    '''

    ztotalsteps=np.int(np.rint(z_field_of_view/zsteps/2))
    rtotalsteps=np.int(np.rint(field_of_view/rsteps*2**0.5/2))

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

    x2,y2=(np.int(np.rint(field_of_view/rsteps)-2),np.int(np.rint(field_of_view/rsteps))-2)#el -2 es para que no haya problemas con el paso a polare
    exx2=np.zeros((x2,y2),dtype=complex)
    eyx2=np.zeros((x2,y2),dtype=complex)
    ezx2=np.zeros((x2,y2),dtype=complex)
    exy2=np.zeros((x2,y2),dtype=complex)
    eyy2=np.zeros((x2,y2),dtype=complex)
    ezy2=np.zeros((x2,y2),dtype=complex)
    zz=ztotalsteps + int(np.rint(zp0/z_field_of_view*2*ztotalsteps))  #zz signals to the row of kz=kz0 in each II
    for xx in range(x2):
        for yy in range(y2):
            xcord=xx - np.rint(2*rtotalsteps /2**0.5)/2#not sure of multipliing by 2 and dividing by 2 outside the int, i thought it was to be sure to get the 0,0 at xx=np.rint(2*rtotalsteps /np.sqrt(2))/2
            ycord=-yy + np.rint(2*rtotalsteps /2**0.5)/2-1
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
    


