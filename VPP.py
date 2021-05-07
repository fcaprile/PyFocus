# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:18:12 2020

@author: ferchi
"""


import numpy as np
from scipy.special import jv
from complex_quadrature import complex_quadrature
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy import interpolate

def VPP_integration(alpha,n,f,w0,wavelength,rsteps,zsteps,field_of_view,z_field_of_view):    
    '''
    Generate the II matrixes, which are the result of the integration for different positions along the radius and z
    This matrixes are later used to calculate the field
    
    wavelength is given in the medium (equals wavelength in vacuum/n)
    '''
    ztotalsteps=np.int(np.rint(z_field_of_view/zsteps/2))
    rtotalsteps=np.int(np.rint(field_of_view/rsteps*2**0.5/2)) #the actual field of view of the X axis in the XZ plane will be field_of_view*2**0.5
    
    gaussian=lambda theta:np.exp(-(np.sin(theta)*f/w0)**2)#inciding gaussian beam's amplitude
    
    I1=np.zeros((ztotalsteps,rtotalsteps),dtype=complex)
    I2=np.copy(I1)
    I3=np.copy(I1)
    I4=np.copy(I1)
    I5=np.copy(I1)
    
    fun4=lambda theta: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1 + np.cos(theta))*jv(1,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun5=lambda theta: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*jv(1,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun6=lambda theta: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*jv(3,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun7=lambda theta: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)**2*jv(0,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun8=lambda theta: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)**2*jv(2,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
        
    for zz in tqdm(range(ztotalsteps),desc='VPP field calculation'):
        for rr in range(rtotalsteps):
            kz=zz*2*np.pi/wavelength/ztotalsteps*z_field_of_view/2 
            kr=rr*2*np.pi/wavelength/rtotalsteps*field_of_view/2*2**0.5
            I1[zz,rr]=complex_quadrature(fun4,0,alpha)[0]
            I2[zz,rr]=complex_quadrature(fun5,0,alpha)[0]
            I3[zz,rr]=complex_quadrature(fun6,0,alpha)[0]
            I4[zz,rr]=complex_quadrature(fun7,0,alpha)[0]
            I5[zz,rr]=complex_quadrature(fun8,0,alpha)[0]

    #since the calculus is the same for all 4 cuadrants, we calculate only one and now we mirror it ugammade-down
    II1=np.vstack((np.flipud(np.conj(I1)),I1[1:,:]))
    II2=np.vstack((np.flipud(np.conj(I2)),I2[1:,:]))
    II3=np.vstack((np.flipud(np.conj(I3)),I3[1:,:]))
    II4=np.vstack((np.flipud(np.conj(I4)),I4[1:,:]))
    II5=np.vstack((np.flipud(np.conj(I5)),I5[1:,:]))
    return II1,II2,II3,II4,II5

def VPP_fields(II1,II2,II3,II4,II5,wavelength,I0,gamma,beta,rsteps,zsteps,field_of_view,z_field_of_view,phip0,n,f,zp0):
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
    exx=a1*np.hstack((- np.fliplr(II1)*np.exp(1j*phip) + 0.5*np.fliplr(II2)*np.exp(- 1j*phip) - 0.5*np.fliplr(II3)*np.exp(3j*phip),II1[:,1:rtotalsteps-1]*np.exp(1j*phip) - 0.5*II2[:,1:rtotalsteps-1]*np.exp(- 1j*phip) + 0.5*II3[:,1:rtotalsteps-1]*np.exp(3j*phip)))
    eyx=-0.5*1j*a1*np.hstack((- np.fliplr(II2)*np.exp(- 1j*phip) - np.fliplr(II3)*np.exp(3j*phip),II2[:,1:rtotalsteps-1]*np.exp(- 1j*phip) + II3[:,1:rtotalsteps-1]*np.exp(3j*phip)))
    ezx=a1*1j*np.hstack((np.fliplr(II4) - np.fliplr(II5)*np.exp(2j*phip),II4[:,1:rtotalsteps-1] - II5[:,1:rtotalsteps-1]*np.exp(2j*phip)))
    
    exy=- 0.5*a2*1j*np.hstack((- np.fliplr(II2)*np.exp(- 1j*phip) - np.fliplr(II3)*np.exp(3j*phip),II2[:,1:rtotalsteps-1]*np.exp(- 1j*phip) + II3[:,1:rtotalsteps-1]*np.exp(3j*phip)))
    eyy=a2*np.hstack((- np.fliplr(II1)*np.exp(1j*phip) - 0.5*np.fliplr(II2)*np.exp(- 1j*phip) + 0.5*np.fliplr(II3)*np.exp(3j*phip),II1[:,1:rtotalsteps-1]*np.exp(1j*phip) + 0.5*II2[:,1:rtotalsteps-1]*np.exp(- 1j*phip) - 0.5*II3[:,1:rtotalsteps-1]*np.exp(3j*phip)))
    ezy=- a2*np.hstack((np.fliplr(II4) + np.fliplr(II5)*np.exp(2j*phip),II4[:,1:rtotalsteps-1] +II5[:,1:rtotalsteps-1]*np.exp(2j*phip)))

    Ex=exx + exy
    Ey=eyx + eyy
    Ez=ezx + ezy
    
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
            exx2[yy,xx]=a1*(II1[zz,rp]*np.exp(1j*phip) - 0.5*II2[zz,rp]*np.exp(-1j*phip) + 0.5*II3[zz,rp]*np.exp(3j*phip))
            eyx2[yy,xx]=- 0.5*a1*1j*(II2[zz,rp]*np.exp(- 1j*phip) + II3[zz,rp]*np.exp(3j*phip))
            ezx2[yy,xx]=a1*1j*(II4[zz,rp] - II5[zz,rp]*np.exp(2j*phip))
            exy2[yy,xx]=-0.5*a2*1j*(II2[zz,rp]*np.exp(- 1j*phip) +II3[zz,rp]*np.exp(3j*phip))
            eyy2[yy,xx]=a2*(II1[zz,rp]*np.exp(1j*phip) + 0.5*II2[zz,rp]*np.exp(- 1j*phip) - 0.5*II3[zz,rp]*np.exp(3j*phip))
            ezy2[yy,xx]=- a2*(II4[zz,rp] + II5[zz,rp]*np.exp(2j*phip))
    Ex2=exx2 + exy2
    Ey2=eyx2 + eyy2
    Ez2=ezx2 + ezy2

    return Ex,Ey,Ez,Ex2,Ey2,Ez2


    

def VPP_fraunhofer(gamma=45,beta=-90,steps=500,h=5,L=100,I_0=1,Lambda=0.00075,FOV=20000,radius=10,limit=2000,div=1,plot=True,save=False,folder='',figure_name=''):
    '''
    Calculate and plot the field inciding on the lens by Fraunhofer's difraction formula
    Returns E_rho, the inciding amplitude along the radial coordinate for an x polarized beam
    Returns Ex and Ey, the x and y components of this amplitude, in a matrix over the x and y coordinates so it can be ploted easily
    plot=True plots the inciding field's intensity and amplitude'
    wavelength is given in the medium (equals wavelength in vacuum/n)
    '''
    
    def complex_quadrature(func, a, b, eabs=1.49e-08, erel=1.49e-08,lim=50):
        def real_func(x):
            return np.real(func(x))
        def imag_func(x):
            return np.imag(func(x))
        real_integral = quad(real_func, a, b, epsabs=eabs, epsrel=erel,limit=lim)
        imag_integral = quad(imag_func, a, b, epsabs=eabs, epsrel=erel,limit=lim)
        return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
        
    # calculating the rho values in wich to integrate
    rmax=FOV
    k=2*np.pi/Lambda 
    rvalues=np.linspace(0,rmax*2**0.5,steps)
    
    
    #E0x, E0y are the polarization elements of the incident wave
    #E_xy is the amplitude distribution along the xy plane for the incident wave (asumed constant but can be switched for a gaussian)
    E_xy=lambda rho: np.exp(-(rho/radius)**2) 
    Ax=I_0**0.5*np.cos(np.pi*gamma/180)  
    Ay=I_0**0.5*np.sin(np.pi*gamma/180)*np.exp(1j*np.pi*beta/180)
    
    fun=lambda rho: E_xy(rho)*rho*np.exp(1j*np.pi/Lambda/L*rho**2)*jv(1,k/L*rho*rhop)
           
    tot=len(rvalues)
    
    Int=np.zeros(tot,dtype=complex)
     
    for i in tqdm(range(tot),desc='Calculating field at the objective'):
        rhop=rvalues[i]
        for l in range(div):
            Int[i]+=complex_quadrature(fun,h*l/div,h*(l+1)/div,lim=limit)[0]
    
    #interpolating the Integration for values of rho:
    Int_interpolated=interpolate.interp1d(rvalues,Int,kind='cubic')    

    #calculating the field along the X, Y plane
    xmax=FOV/2
    xyvalues=np.linspace(-xmax,xmax,int(np.rint(steps/2**0.5/4)))
    tot_xy=len(xyvalues)
    #Ei (i= x,y,z) are the fieds to calculate
    Ex=np.zeros((tot_xy,tot_xy),dtype=complex)
    Ey=np.copy(Ex)
    
    E_fun=lambda rho:2*np.pi*np.exp(1j*k*(L+rho**2/2/L))/Lambda/L*Int_interpolated(rho)


    for i,xp in enumerate(xyvalues): 
        for j,yp in enumerate(xyvalues): 
            rhop=(xp**2+yp**2)**0.5
            phip=np.arctan2(yp,xp)
            Ex[j,i]=Ax*E_fun(rhop)*np.exp(1j*phip)
            Ey[j,i]=Ay*E_fun(rhop)*np.exp(1j*phip)
    
    
    x,y=np.shape(Ex)
    Ifield=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            Ifield[j,i]=np.real(Ex[j,i]*np.conj(Ex[j,i])+Ey[j,i]*np.conj(Ey[j,i]))
        
    # xy_res=xyvalues[1]-xyvalues[0]
    # Int_final=np.sum(Ifield)*(xy_res)**2
    # transmission=Int_final/(np.pi*h**2*I_0)        
    # print('Transmission= ',transmission)                
    
    #intensity and fit plot
    plt.rcParams['font.size']=20#tamaño de fuente
    fig1, (ax1, ax2) = plt.subplots(num=str(figure_name)+': Incident intensity',figsize=(12, 5), ncols=2)
    fig1.suptitle('Field at objective')

    ax1.set_title('Intensity')
    pos=ax1.imshow(Ifield,extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.axis('square')
    cbar1= fig1.colorbar(pos, ax=ax1)
    cbar1.ax.set_ylabel('Intensity (kW/cm\u00b2)')

    
    x2=np.shape(Ifield)[0]
    ax2.set_title(' Intensity along x')
    ax2.plot(np.linspace(-xmax,xmax,x2),Ifield[int(x2/2),:])
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('Intensity  (kW/cm\u00b2)')  
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.80)
    #amplitude and phase plot 
    #ex
    fig2, ((ax_x1,ax_y1),(ax_x2,ax_y2)) = plt.subplots(num=str(figure_name)+': Incident amplitude',figsize=(12, 8),nrows=2, ncols=2)
    fig2.suptitle('Field at objective')
    ax_x1.set_title('ex amplitude')
    pos_x1=ax_x1.imshow(np.abs(Ex),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_x1.set_xlabel('x (mm)')
    ax_x1.set_ylabel('y (mm)')
    ax_x1.axis('square')
    cbar_1_1=fig2.colorbar(pos_x1, ax=ax_x1)
    cbar_1_1.ax.set_ylabel('Relative amplitude')
    
    ax_x2.set_title('ex phase')
    pos_x2=ax_x2.imshow(np.angle(Ex),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_x2.set_xlabel('x (mm)')
    ax_x2.set_ylabel('y (mm)')
    ax_x2.axis('square')
    cbar_1_1=fig2.colorbar(pos_x2, ax=ax_x2)
    cbar_1_1.ax.set_ylabel('Angle (Radians)')
    
    #ey
    ax_y1.set_title('ey amplitude')
    pos_y1=ax_y1.imshow(np.abs(Ey),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_y1.set_xlabel('x (mm)')
    ax_y1.set_ylabel('y (mm)')
    ax_y1.axis('square')
    cbar_1_1=fig2.colorbar(pos_y1, ax=ax_y1)
    cbar_1_1.ax.set_ylabel('Relative amplitude')
    
    ax_y2.set_title('ey phase')
    ax_y2.axis('square')
    pos_y2=ax_y2.imshow(np.angle(Ey),extent=[-xmax,xmax,-xmax,xmax], interpolation='none', aspect='auto')
    ax_y2.set_xlabel('x (mm)')
    ax_y2.set_ylabel('y (mm)')
    ax_y2.axis('square')
    cbar_1_1=fig2.colorbar(pos_y2, ax=ax_y2)
    cbar_1_1.ax.set_ylabel('Angle (Radians)')
    
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.88)
    '''
    if figures are to be saved automatically:
    fig1.savefig(folder+'Intensity h = '+str(h)+', radius = '+str(radius)+' pre focus.png')
    fig2.savefig(folder+'Amplitude h = '+str(h)+', radius = '+str(radius)+' pre focus.png')
    '''    
    return E_fun,Ex,Ey

def VPP_integration_with_propagation(alpha,n,f,radius_VPP,wavelength,zp0,zsteps,rsteps,field_of_view,laser_width,E_rho,div):    
    '''
    Given the inciding field E_rho, which only depends on the radial coordinate, generate the I matrixes, which are the same as in VPP_integration
    Since the calculus takes a long time, only the field along the XY plane is calculated
    wavelength is given in the medium (equals wavelength in vacuum/n)
    '''
    E_theta=lambda theta: E_rho(np.sin(theta)*f*10**-6)     #10**-6 is for passage from mm to nm
    
    rtotalsteps=np.int(np.rint(field_of_view/rsteps*2**0.5/2))
    
    I1=np.zeros(rtotalsteps,dtype=complex)
    I2=np.copy(I1)
    I3=np.copy(I1)
    I4=np.copy(I1)
    I5=np.copy(I1)
    
    fun4=lambda theta: E_theta(theta)*np.cos(theta)**0.5*np.sin(theta)*(1 + np.cos(theta))*jv(1,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun5=lambda theta: E_theta(theta)*np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*jv(1,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun6=lambda theta: E_theta(theta)*np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*jv(3,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun7=lambda theta: E_theta(theta)*np.cos(theta)**0.5*np.sin(theta)**2*jv(0,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun8=lambda theta: E_theta(theta)*np.cos(theta)**0.5*np.sin(theta)**2*jv(2,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    
    # alpha_VPP=np.arctan(radius_VPP/(f*wavelength))#since radius_VPP is given in nm, i translate f to nm    
    for rr in tqdm(range(rtotalsteps),desc='Focal plane field calculation'):
        kz=zp0*2*np.pi/wavelength 
        kr=rr*2*np.pi/wavelength/rtotalsteps*field_of_view/2*2**0.5
        for l in range(div):
            I1[rr]+=complex_quadrature(fun4,alpha*l/div,alpha*(l+1)/div)[0]
            I2[rr]+=complex_quadrature(fun5,alpha*l/div,alpha*(l+1)/div)[0]
            I3[rr]+=complex_quadrature(fun6,alpha*l/div,alpha*(l+1)/div)[0]
            I4[rr]+=complex_quadrature(fun7,alpha*l/div,alpha*(l+1)/div)[0]
            I5[rr]+=complex_quadrature(fun8,alpha*l/div,alpha*(l+1)/div)[0]

    return I1,I2,I3,I4,I5


def VPP_fields_with_propagation(I1,I2,I3,I4,I5,wavelength,I0,gamma,beta,rsteps,zsteps,field_of_view,phip0,n,f,zp0):
    '''
    Given the I matrixes calculate the field on the focus
    Since the calculus takes a long time, only the field along the XY plane is calculated
    parameter phip0 has no purpose, is only left to have the same variables for the functions
    wavelength is given in the medium (equals wavelength in vacuum/n)
    '''
    rtotalsteps=np.int(np.rint(field_of_view/rsteps*2**0.5/2))

    def cart2pol(x,y):    
        r = (x**2+y**2)**0.5
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
        

    x,y=(np.int(np.rint(field_of_view/rsteps)-2),np.int(np.rint(field_of_view/rsteps))-2)#el -2 es para que no haya problemas con el paso a polare
    exx=np.zeros((x,y),dtype=complex)
    eyx=np.zeros((x,y),dtype=complex)
    ezx=np.zeros((x,y),dtype=complex)
    exy=np.zeros((x,y),dtype=complex)
    eyy=np.zeros((x,y),dtype=complex)
    ezy=np.zeros((x,y),dtype=complex)

    for xx in range(x):
        for yy in range(y):
            xcord=xx - np.rint(2*rtotalsteps /np.sqrt(2))/2#not sure of multiplIng by 2 and dividing by 2 outside the int, i thought it was to be sure to get the 0,0 at xx=np.rint(2*rtotalsteps /np.sqrt(2))/2
            ycord=yy - np.rint(2*rtotalsteps /np.sqrt(2))/2
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

    return Ex,Ey,Ez


