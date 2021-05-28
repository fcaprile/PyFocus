# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:40:54 2020

@author: Fer Caprile

Functions for simulation of the diffraction obtained by an arbitrary phase mask
"""

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
from tmm_core import *


def interphase_custom_mask_focus_field_XY(n_list,d_list,ex_lens,ey_lens,alpha,h,Lambda,z_interface,zp0,resolution_focus,resolution_theta,resolution_phi,FOV_focus,countdown=True,x0=0,y0=0):
    '''
    2D integration to calculate the field at the focus of a high aperture lens with an interphase
    ex_lens,ey_lens are the x and y component of the inciding field
    Calculates the field on the XY focal plane.
    
    resolution_focus is the resolution for the field at the focus, the same for x and y
    resolution_theta,resolution_phi is the resolution for the 2D calculus (must be the same as the sie of ex_lens and ey_lens) 
    
    wavelength is given in the medium (equals wavelength in vacuum/n)
    countdown=True means you are only running this fuction once and you want to see te time elapsed and expected to finish the calculation
    
    x0 and y0 are used for calculating the field centered at an x0, y0 position
    '''


    n1=n_list[0]#first medium
    n2=n_list[-1]#last medium
    
    if countdown==True:
        print('Calculating field at the focal plane:')
        time.sleep(0.5)
    Lambda*=10**6 #from mm to nm
    focus=h/np.sin(alpha)*10**6 #from mm to nm
        
    #The Y component of incident field must be evaluated at phi-pi/2, which is equivalent to moving the rows of the matrix    
    def rotate_90º(matrix):
        a,b=np.shape(matrix)       
        aux=np.zeros((a,b),dtype=complex)        
        for i in range(a):
            aux[i-int(a/4),:]=matrix[i,:]
        return aux

    ey_lens=rotate_90º(ey_lens)

    #Functions for computing the reflection and transmitinos coeficients between 2 interfaces. Subindexes _i meant incident and _t meand transmited  
    #given incident angle (theta_i), compute the reflection coeficients:
    n12=n1/n2 #to save computing time, might not be worth it
    '''
    #functions to integrate: Focused field without interface (Ef)
    fun1=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun2=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun3=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun4=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun5=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun6=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    
    k1=Lambda
    k2=Lambda*n21
    kz_r=zp0*2*np.pi/k1
    kz_t=zp0*2*np.pi/k2
    #functions to integrate: Reflected field (Er)
    fun1_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(-rs_i(theta)*np.sin(phi)**2+rp_i(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun2_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(rs_i(theta) + rp_i(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun3_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*rp_i(theta)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun4_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(rs_i(theta) + rp_i(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))
    fun5_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(-rs_i(theta)*np.sin(phi)**2+rp_i*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))
    fun6_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*rp_i(theta)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))

    k2=Lambda*n21
    #functions to integrate: Transmited field (Et)
    k1_minus_k2=k1-k2
    fun1_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(ts_t(theta)*np.sin(phi)**2+tp_t(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun2_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(-ts_t(theta) + tp_t(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun3_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*tp_t(theta)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun4_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(-ts_t(theta) + tp_t(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    fun5_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(ts_t(theta)*np.sin(phi)**2+tp_t(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    fun6_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*tp_t(theta)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    
    '''
    #The total field is Ef+Er for z<zp0 and Et for z>zp0
        
    #2D trapezoidal method weight:
    h_theta=alpha/resolution_theta
    h_phi=2*np.pi/resolution_phi
    weight_trapezoid_rho=np.zeros(resolution_theta)+h_theta
    weight_trapezoid_rho[0]=h_theta/2
    weight_trapezoid_rho[-1]=h_theta/2
    weight_trapezoid_phi=np.zeros(resolution_phi)+h_phi
    weight_trapezoid_phi[0]=h_phi/2
    weight_trapezoid_phi[-1]=h_phi/2
    weight_trapezoid=weight_trapezoid_rho*np.vstack(weight_trapezoid_phi)
    
    #define positions in which to calculate the field:    
    xmax=FOV_focus/2
    x_values=np.linspace(-xmax+x0,xmax+x0,resolution_focus)
    y_values=np.linspace(xmax+y0,-xmax+y0,resolution_focus)
    ex=np.zeros((resolution_focus,resolution_focus),dtype=complex)
    ey=np.copy(ex)
    ez=np.copy(ex)
    
    
    #define divisions for the integration:
    theta_values=np.linspace(0,alpha,resolution_theta)    
    phi_values=np.linspace(0,2*np.pi,resolution_phi) 
    theta,phi=np.meshgrid(theta_values,phi_values)
    
    #now begins the integration, in order to save computing time i do the trigonometric functions separatedly and save the value into an auxiliar variable. This reduces computing time up to 8 times
    cos_theta=np.cos(theta)
    cos_theta_sqrt=cos_theta**0.5
    sin_theta=np.sin(theta)
    cos_phi=np.cos(phi)
    cos_phi_square=cos_phi**2
    sin_phi=np.sin(phi)
    sin_phi_square=sin_phi**2
    
    #For integration of the field without interphace (Ef):
    k1=2*np.pi/Lambda*n1
    k2=k1*n2/n1
    prefactor_general=cos_theta_sqrt*sin_theta*k1
    prefactor_x=prefactor_general*(cos_theta+(1-cos_theta)*sin_phi_square)
    prefactor_y=prefactor_general*(1-cos_theta)*cos_phi*sin_phi
    prefactor_z=prefactor_general*sin_theta*cos_phi
    
    Axx=-prefactor_x*ex_lens*weight_trapezoid
    Axy=-prefactor_y*ex_lens*weight_trapezoid
    Axz=-prefactor_z*ex_lens*weight_trapezoid

    Ayx=-prefactor_y*ey_lens*weight_trapezoid
    Ayy=prefactor_x*ey_lens*weight_trapezoid
    Ayz=-prefactor_z*ey_lens*weight_trapezoid
    
    #Calculus of the refraction and transmition coeficients using german's code
    rs_i_theta=np.zeros((resolution_phi,resolution_theta),dtype='complex')
    rp_i_theta=np.copy(rs_i_theta)
    ts_t_theta=np.copy(rs_i_theta)
    tp_t_theta=np.copy(rs_i_theta)
    theta_values=np.linspace(0,alpha,resolution_theta)    
    for i in range(resolution_theta):
        theta_val=theta_values[i]
        tmm_p=coh_tmm('p', n_list, d_list, theta_val, Lambda)
        tmm_s=coh_tmm('s', n_list, d_list, theta_val, Lambda)
        rs_i_theta[:,i]=tmm_s['r']
        rp_i_theta[:,i]=tmm_p['r']
        ts_t_theta[:,i]=tmm_s['t']
        tp_t_theta[:,i]=tmm_p['t']
    #debuging: auxiliar_plot_something(np.abs(rs_i_theta),0,1,0,1)
    #For integration of the reflected and transmited fields (Er and Et):
    prefactor_x_r=-prefactor_general*(-rs_i_theta*sin_phi_square+rp_i_theta*cos_phi_square*cos_theta)
    prefactor_y_r=-prefactor_general*(rs_i_theta+rp_i_theta*cos_theta)*cos_phi*sin_phi
    prefactor_z_r=prefactor_general*rp_i_theta*sin_theta*cos_phi
    
    phase_z_r=np.exp(2*1j*k1*np.cos(theta)*z_interface)
    
    Axx_r=-prefactor_x_r*ex_lens*weight_trapezoid
    Axy_r=prefactor_y_r*ex_lens*weight_trapezoid
    Axz_r=prefactor_z_r*ex_lens*weight_trapezoid

    Ayx_r=phase_z_r*prefactor_y_r*ey_lens*weight_trapezoid
    Ayy_r=phase_z_r*prefactor_x_r*ey_lens*weight_trapezoid
    Ayz_r=phase_z_r*prefactor_z_r*ey_lens*weight_trapezoid
 
    #switching to complex angles in order to compute the transmited complex angles:
    theta_values_complex=np.linspace(0,alpha,resolution_theta,dtype='complex')    
    phi_values_complex=np.linspace(0,2*np.pi,resolution_phi,dtype='complex') 
    theta_complex,phi_complex=np.meshgrid(theta_values_complex,phi_values_complex)

    sin_theta_complex=np.sin(theta_complex)

    cos_theta_t=(1-(n12*sin_theta_complex)**2)**0.5
    sin_theta_t=n12*sin_theta #snell
    prefactor_general_t=(cos_theta)**0.5*sin_theta*k1

    prefactor_x_t=prefactor_general_t*(ts_t_theta*sin_phi_square+tp_t_theta*cos_phi_square*cos_theta_t)
    prefactor_y_t=prefactor_general_t*(-ts_t_theta+tp_t_theta*cos_theta_t)*cos_phi*sin_phi
    prefactor_z_t=-prefactor_general_t*tp_t_theta*sin_theta_t*cos_phi
    
    phase_z_t=np.exp(1j*z_interface*(k2*cos_theta_t+k1*cos_theta))
    
    Axx_t=-phase_z_t*prefactor_x_t*ex_lens*weight_trapezoid
    Axy_t=phase_z_t*prefactor_y_t*ex_lens*weight_trapezoid
    Axz_t=phase_z_t*prefactor_z_t*ex_lens*weight_trapezoid

    Ayx_t=phase_z_t*prefactor_y_t*ey_lens*weight_trapezoid
    Ayy_t=phase_z_t*prefactor_x_t*ey_lens*weight_trapezoid
    Ayz_t=phase_z_t*prefactor_z_t*ey_lens*weight_trapezoid

    kz=zp0*k1
    kz_r=zp0*k1

    phase_kz=np.exp(1j*cos_theta*kz)
    phase_kz_r=np.exp(-1j*cos_theta*kz_r)
    kz_t=zp0*k2
    phase_kz_t=np.exp(1j*cos_theta_t*kz_t)

    if countdown==True:
        for i in tqdm(range(resolution_focus)):
            x=x_values[i]
            for j,y in enumerate(y_values):#idea, rotar en phi es correr las columnas de la matriz ex, ey
                rhop=(x**2+y**2)**0.5
                phip=np.arctan2(y,x)
                kr=rhop*k1
                sin_theta_kr=sin_theta*kr
                
                phase_rho_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip)))
                phase_rho_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip)))
                
                if z_interface<=zp0:
                    phase_inc_x=phase_rho_x*phase_kz_t      #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz_t      #phase for the Y incident component of the transmited field
                    ex[j,i]=np.sum(Axx_t*phase_inc_x)+np.sum(Ayx_t*phase_inc_y)
                    ey[j,i]=np.sum(Axy_t*phase_inc_x)+np.sum(Ayy_t*phase_inc_y)
                    ez[j,i]=np.sum(Axz_t*phase_inc_x)+np.sum(Ayz_t*phase_inc_y)
                else:
                    phase_inc_x=phase_rho_x*phase_kz        #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz        #phase for the Y incident component of the transmited field
                    phase_inc_x_r=phase_rho_x*phase_kz_r    #phase for the X incident component of the reflected field
                    phase_inc_y_r=phase_rho_y*phase_kz_r    #phase for the Y incident component of the reflected field
                    ex[j,i]=np.sum(Axx*phase_inc_x+Axx_r*phase_inc_x_r)+np.sum(Ayx*phase_inc_y+Ayx_r*phase_inc_y_r)
                    ey[j,i]=np.sum(Axy*phase_inc_x+Axy_r*phase_inc_x_r)+np.sum(Ayy*phase_inc_y+Ayy_r*phase_inc_y_r)
                    ez[j,i]=np.sum(Axz*phase_inc_x+Axz_r*phase_inc_x_r)+np.sum(Ayz*phase_inc_y+Ayz_r*phase_inc_y_r)
    else:
        for i,x in enumerate(x_values):
            for j,y in enumerate(y_values):
                rhop=(x**2+y**2)**0.5
                phip=np.arctan2(y,x)
                kr=rhop*k
                sin_theta_kr=sin_theta*kr
                
                phase_rho_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip)))
                phase_rho_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip)))
                
                if z_interface<=zp0:
                    phase_inc_x=phase_rho_x*phase_kz_t      #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz_t      #phase for the Y incident component of the transmited field
                    ex[j,i]=np.sum(Axx_t*phase_inc_x)+np.sum(Ayx_t*phase_inc_y)
                    ey[j,i]=np.sum(Axy_t*phase_inc_x)+np.sum(Ayy_t*phase_inc_y)
                    ez[j,i]=np.sum(Axz_t*phase_inc_x)+np.sum(Ayz_t*phase_inc_y)
                else:
                    phase_inc_x=phase_rho_x*phase_kz        #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz        #phase for the Y incident component of the transmited field
                    phase_inc_x_r=phase_rho_x*phase_kz_r    #phase for the X incident component of the reflected field
                    phase_inc_y_r=phase_rho_y*phase_kz_r    #phase for the Y incident component of the reflected field
                    ex[j,i]=np.sum(Axx*phase_inc_x+Axx_r*phase_inc_x_r)+np.sum(Ayx*phase_inc_y+Ayx_r*phase_inc_y_r)
                    ey[j,i]=np.sum(Axy*phase_inc_x+Axy_r*phase_inc_x_r)+np.sum(Ayy*phase_inc_y+Ayy_r*phase_inc_y_r)
                    ez[j,i]=np.sum(Axz*phase_inc_x+Axz_r*phase_inc_x_r)+np.sum(Ayz*phase_inc_y+Ayz_r*phase_inc_y_r)
        
    ex*=-1j*focus/2/np.pi*np.exp(-1j*k1*focus)
    ey*=1j*focus/2/np.pi*np.exp(-1j*k1*focus)
    ez*=1j*focus/2/np.pi*np.exp(-1j*k1*focus)
    
    return ex,ey,ez

def interphase_custom_mask_focus_field_XZ_XY(n_list,d_list,ex_lens,ey_lens,alpha,h,Lambda,z_interface,z_FOV,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,FOV_focus,x0=0,y0=0,z0=0,plot_plane='X'):
    '''
    2D integration to calculate the field at the focus of a high aperture lens with an interphase
    ex_lens,ey_lens are the x and y component of the inciding field
    Calculates the field on the XY focal plane and the XZ plane.
    
    resolution_focus is the resolution for the field at the focus, the same for x and y
    resolution_theta,resolution_phi is the resolution for the 2D calculus (must be the same as the sie of ex_lens and ey_lens) 
    
    wavelength is given in the medium (equals wavelength in vacuum/n)
    countdown=True means you are only running this fuction once and you want to see te time elapsed and expected to finish the calculation
    
    x0 and y0 are used for calculating the field centered at an x0, y0 position
    '''


    n1=n_list[0]
    n2=n_list[-1]
    #XY plane: 
    Ex_XY,Ey_XY,Ez_XY=interphase_custom_mask_focus_field_XY(n_list,d_list,ex_lens,ey_lens,alpha,h,Lambda,z_interface,zp0,resolution_focus,resolution_theta,resolution_phi,FOV_focus,True,x0,y0)
    
    #XZ plane:
    if int(resolution_z%2)==0:
        resolution_z+=1    #make the middle coordinate on Z be Z=0
        
    Lambda*=10**6               #passage from mm to nm
    focus=h/np.sin(alpha)*10**6 #passage from mm to nm
        
    #The Y component of incident field must be evaluated at phi-pi/2, which is equivalent to moving the rows of the matrix    
    def rotate_90º(matrix):
        a,b=np.shape(matrix)       
        aux=np.zeros((a,b),dtype=complex)        
        for i in range(a):
            aux[i-int(a/4),:]=matrix[i,:]
        return aux

    ey_lens=rotate_90º(ey_lens)

    '''
    #functions to integrate: Focused field without interface (Ef)
    fun1=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun2=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun3=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz))
    fun4=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun5=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)*(np.cos(theta) + (1 - np.cos(theta))*np.sin(phi)**2)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    fun6=lambda phi,theta: np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz))
    
    k1=2*np.pi/Lambda
    kz_r=zp0*2*np.pi/k1
    #functions to integrate: Reflected field (Er)
    fun1_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(-rs_i(theta)*np.sin(phi)**2+rp_i(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun2_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(rs_i(theta) + rp_i(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun3_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*rp_i(theta)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) - np.cos(theta)*kz_r))
    fun4_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(rs_i(theta) + rp_i(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))
    fun5_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(-rs_i(theta)*np.sin(phi)**2+rp_i*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))
    fun6_i=lambda phi,theta: -np.exp(2*1j*k1*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*rp_i(theta)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) - np.cos(theta)*kz_r))

    #functions to integrate: Transmited field (Et), would have to be integrated for theta_t (the angle of the transmited field) wich can be complex
    k2=k1*n2/n1
    k1_minus_k2=k1-k2
    kz_t=zp0*2*np.pi/k2  
    fun1_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(ts_t(theta)*np.sin(phi)**2+tp_t(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun2_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(-ts_t(theta) + tp_t(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun3_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*tp_t(theta)*np.exp(1j*(np.sin(theta)*kr*np.cos(phi - phip) + np.cos(theta)*kz_t))
    fun4_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(-ts_t(theta) + tp_t(theta)*np.cos(theta))*np.cos(phi)*np.sin(phi)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    fun5_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)*(ts_t(theta)*np.sin(phi)**2+tp_t(theta)*np.cos(phi)**2*np.cos(theta))*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    fun6_t=lambda phi,theta: np.exp(2*1j*k1_minus_k2*np.cos(theta)*z_interface)*np.cos(theta)**0.5*np.sin(theta)**2*np.cos(phi)*tp_t(theta)*np.exp(1j*(-np.sin(theta)*kr*np.sin(phi - phip) + np.cos(theta)*kz_t))
    '''

    #The total field is Ef+Er for z<zp0 and Et for z>zp0
        
    #2D trapezoidal method weight:
    h_theta=alpha/resolution_theta
    h_phi=2*np.pi/resolution_phi
    weight_trapezoid_rho=np.zeros(resolution_theta)+h_theta
    weight_trapezoid_rho[0]=h_theta/2
    weight_trapezoid_rho[-1]=h_theta/2
    weight_trapezoid_phi=np.zeros(resolution_phi)+h_phi
    weight_trapezoid_phi[0]=h_phi/2
    weight_trapezoid_phi[-1]=h_phi/2
    weight_trapezoid=weight_trapezoid_rho*np.vstack(weight_trapezoid_phi)
       
    
    #define divisions for the integration:
    theta_values=np.linspace(0,alpha,resolution_theta)    
    phi_values=np.linspace(0,2*np.pi,resolution_phi) 
    theta,phi=np.meshgrid(theta_values,phi_values)
    
    #now begins the integration, in order to save computing time i do the trigonometric functions separatedly and save the value into an auxiliar variable. This reduces computing time up to 8 times
    cos_theta=np.cos(theta)
    cos_theta_sqrt=cos_theta**0.5
    sin_theta=np.sin(theta)
    cos_phi=np.cos(phi)
    cos_phi_square=cos_phi**2
    sin_phi=np.sin(phi)
    sin_phi_square=sin_phi**2
    
    #For integration of the field without interphace (Ef):
    k1=2*np.pi/Lambda*n1
    k2=k1*n2/n1
    prefactor_general=cos_theta_sqrt*sin_theta*k1
    prefactor_x=prefactor_general*(sin_phi_square+cos_phi_square*cos_theta)
    prefactor_y=prefactor_general*(-1+cos_theta)*cos_phi*sin_phi
    prefactor_z=-prefactor_general*sin_theta*cos_phi
    
    Axx=-prefactor_x*ex_lens*weight_trapezoid
    Axy=prefactor_y*ex_lens*weight_trapezoid
    Axz=prefactor_z*ex_lens*weight_trapezoid

    Ayx=prefactor_y*ey_lens*weight_trapezoid
    Ayy=prefactor_x*ey_lens*weight_trapezoid
    Ayz=prefactor_z*ey_lens*weight_trapezoid

    #Calculus of the refraction and transmition coeficients using german's code
    rs_i_theta=np.zeros((resolution_phi,resolution_theta),dtype='complex')
    rp_i_theta=np.copy(rs_i_theta)
    ts_t_theta=np.copy(rs_i_theta)
    tp_t_theta=np.copy(rs_i_theta)
    theta_values=np.linspace(0,alpha,resolution_theta) 
    reflejado_values=np.zeros(resolution_theta,dtype='complex')
    transmitido_values=np.zeros(resolution_theta,dtype='complex')
    for i in range(resolution_theta):
        theta_val=theta_values[i]
        tmm_p=coh_tmm('p', n_list, d_list, theta_val, Lambda)
        tmm_s=coh_tmm('s', n_list, d_list, theta_val, Lambda)
        rs_i_theta[:,i]=tmm_s['r']
        rp_i_theta[:,i]=tmm_p['r']
        ts_t_theta[:,i]=tmm_s['t']
        tp_t_theta[:,i]=tmm_p['t']
        reflejado_values[i]=tmm_p['r']
        transmitido_values[i]=tmm_p['t']

    #For integration of the reflected and transmited fields (Er and Et):
    prefactor_x_r=prefactor_general*(rs_i_theta*sin_phi_square-rp_i_theta*cos_phi_square*cos_theta)
    prefactor_y_r=prefactor_general*(-rs_i_theta-rp_i_theta*cos_theta)*cos_phi*sin_phi
    prefactor_z_r=-prefactor_general*rp_i_theta*sin_theta*cos_phi
    
    phase_z_r=np.exp(2*1j*k1*np.cos(theta)*z_interface)
    
    Axx_r=-prefactor_x_r*ex_lens*weight_trapezoid
    Axy_r=prefactor_y_r*ex_lens*weight_trapezoid
    Axz_r=prefactor_z_r*ex_lens*weight_trapezoid

    Ayx_r=phase_z_r*prefactor_y_r*ey_lens*weight_trapezoid
    Ayy_r=phase_z_r*prefactor_x_r*ey_lens*weight_trapezoid
    Ayz_r=phase_z_r*prefactor_z_r*ey_lens*weight_trapezoid
   
    #switching to complex angles in order to compute the transmited complex angles:
    theta_values_complex=np.linspace(0,alpha,resolution_theta,dtype='complex')    
    phi_values_complex=np.linspace(0,2*np.pi,resolution_phi,dtype='complex') 
    theta_complex,phi_complex=np.meshgrid(theta_values_complex,phi_values_complex)

    sin_theta_complex=np.sin(theta_complex)

    cos_theta_t=(1-(n1/n2*sin_theta_complex)**2)**0.5
    sin_theta_t=n1/n2*sin_theta #snell
    prefactor_general_t=(cos_theta)**0.5*sin_theta*k1

    prefactor_x_t=prefactor_general_t*(ts_t_theta*sin_phi_square+tp_t_theta*cos_phi_square*cos_theta_t)
    prefactor_y_t=prefactor_general_t*(-ts_t_theta+tp_t_theta*cos_theta_t)*cos_phi*sin_phi
    prefactor_z_t=-prefactor_general_t*tp_t_theta*sin_theta_t*cos_phi
    
    phase_z_t=np.exp(1j*z_interface*(k2*cos_theta_t+k1*cos_theta))
    
    Axx_t=-phase_z_t*prefactor_x_t*ex_lens*weight_trapezoid
    Axy_t=phase_z_t*prefactor_y_t*ex_lens*weight_trapezoid
    Axz_t=phase_z_t*prefactor_z_t*ex_lens*weight_trapezoid

    Ayx_t=phase_z_t*prefactor_y_t*ey_lens*weight_trapezoid
    Ayy_t=phase_z_t*prefactor_x_t*ey_lens*weight_trapezoid
    Ayz_t=phase_z_t*prefactor_z_t*ey_lens*weight_trapezoid



    #define positions in which to calculate the field:                
    xmax=FOV_focus/2
    x_values=np.linspace(-xmax+x0,xmax+x0,resolution_focus)
    z_values=np.linspace(z_FOV/2+z0,-z_FOV/2+z0,resolution_z)
    
    Ex_XZ=np.zeros((resolution_z,resolution_focus),dtype=complex)
    Ey_XZ=np.copy(Ex_XZ)
    Ez_XZ=np.copy(Ex_XZ)
        
    #define divisions for the integration:
    theta_values=np.linspace(0,alpha,resolution_theta)    
    phi_values=np.linspace(0,2*np.pi,resolution_phi) 
    theta,phi=np.meshgrid(theta_values,phi_values)
    if plot_plane=='X':
        for j in tqdm(range(resolution_z)):
            zp0=z_values[j]
            for i,x in enumerate(x_values):
                rhop=np.abs(x)
                phip=np.arctan2(0,x)
                kr=rhop*k1
                sin_theta_kr=sin_theta*kr       #because of snell's law, this factor will be the same for the reflected and transmited fields
                
                phase_rho_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip)))
                phase_rho_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip)))
                
                if z_interface<=zp0:
                    kz_t=zp0*k2
                    phase_kz_t=np.exp(1j*cos_theta_t*kz_t)
                    phase_inc_x=phase_rho_x*phase_kz_t      #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz_t      #phase for the Y incident component of the transmited field
                    Ex_XZ[j,i]=np.sum(Axx_t*phase_inc_x)+np.sum(Ayx_t*phase_inc_y)
                    Ey_XZ[j,i]=np.sum(Axy_t*phase_inc_x)+np.sum(Ayy_t*phase_inc_y)
                    Ez_XZ[j,i]=np.sum(Axz_t*phase_inc_x)+np.sum(Ayz_t*phase_inc_y)
                else:
                    kz=zp0*k1
                    phase_kz=np.exp(1j*cos_theta*kz)
                    phase_kz_r=np.exp(-1j*cos_theta*kz)
                    phase_inc_x=phase_rho_x*phase_kz        #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz        #phase for the Y incident component of the transmited field
                    phase_inc_x_r=phase_rho_x*phase_kz_r    #phase for the X incident component of the reflected field
                    phase_inc_y_r=phase_rho_y*phase_kz_r    #phase for the Y incident component of the reflected field
                    Ex_XZ[j,i]=np.sum(Axx*phase_inc_x+Axx_r*phase_inc_x_r)+np.sum(Ayx*phase_inc_y+Ayx_r*phase_inc_y_r)
                    Ey_XZ[j,i]=np.sum(Axy*phase_inc_x+Axy_r*phase_inc_x_r)+np.sum(Ayy*phase_inc_y+Ayy_r*phase_inc_y_r)
                    Ez_XZ[j,i]=np.sum(Axz*phase_inc_x+Axz_r*phase_inc_x_r)+np.sum(Ayz*phase_inc_y+Ayz_r*phase_inc_y_r)
    else:
        if plot_plane=='Y':           
            for j in tqdm(range(resolution_z)):
                zp0=z_values[j]
                for i,y in enumerate(x_values):
                    rhop=np.abs(y)
                    phip=np.arctan2(y,0)
                    kr=rhop*k1
                    sin_theta_kr=sin_theta*kr       #because of snell's law, this factor will be the same for the reflected and transmited fields
                    
                    phase_rho_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip)))
                    phase_rho_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip)))
                    
                    if z_interface<=zp0:
                        kz_t=zp0*k2
                        phase_kz_t=np.exp(1j*cos_theta_t*kz_t)
                        phase_inc_x=phase_rho_x*phase_kz_t      #phase for the X incident component of the transmited field
                        phase_inc_y=phase_rho_y*phase_kz_t      #phase for the Y incident component of the transmited field
                        Ex_XZ[j,i]=np.sum(Axx_t*phase_inc_x)+np.sum(Ayx_t*phase_inc_y)
                        Ey_XZ[j,i]=np.sum(Axy_t*phase_inc_x)+np.sum(Ayy_t*phase_inc_y)
                        Ez_XZ[j,i]=np.sum(Axz_t*phase_inc_x)+np.sum(Ayz_t*phase_inc_y)
                    else:
                        kz=zp0*k1
                        phase_kz=np.exp(1j*cos_theta*kz)
                        phase_kz_r=np.exp(-1j*cos_theta*kz)
                        phase_inc_x=phase_rho_x*phase_kz        #phase for the X incident component of the transmited field
                        phase_inc_y=phase_rho_y*phase_kz        #phase for the Y incident component of the transmited field
                        phase_inc_x_r=phase_rho_x*phase_kz_r    #phase for the X incident component of the reflected field
                        phase_inc_y_r=phase_rho_y*phase_kz_r    #phase for the Y incident component of the reflected field
                        Ex_XZ[j,i]=np.sum(Axx*phase_inc_x+Axx_r*phase_inc_x_r)+np.sum(Ayx*phase_inc_y+Ayx_r*phase_inc_y_r)
                        Ey_XZ[j,i]=np.sum(Axy*phase_inc_x+Axy_r*phase_inc_x_r)+np.sum(Ayy*phase_inc_y+Ayy_r*phase_inc_y_r)
                        Ez_XZ[j,i]=np.sum(Axz*phase_inc_x+Axz_r*phase_inc_x_r)+np.sum(Ayz*phase_inc_y+Ayz_r*phase_inc_y_r)
        else:
            print('Options for plot_plane are \'X\' and \'Y\' ')
    Ex_XZ*=1j*focus*np.exp(1j*k1*focus)/2/np.pi
    Ey_XZ*=1j*focus*np.exp(1j*k1*focus)/2/np.pi
    Ez_XZ*=1j*focus*np.exp(1j*k1*focus)/2/np.pi
    
    # plot coeficiente de reflexion
    # plt.rcParams['font.size']=14
    # fig = plt.figure(figsize=(8, 8))
    # spec = fig.add_gridspec(ncols=1, nrows=2)

    # ax2 = fig.add_subplot(spec[1, 0])
    # ax2.plot(np.rad2deg(theta_values),np.angle(reflejado_values,deg=True))
    # ax2.set_xlabel('\u03B8 (º)')
    # ax2.set_ylabel('Fase en la reflectividad')
    
    # ax1 = fig.add_subplot(spec[0, 0],sharex=ax2)
    # ax1.set_title('Coeficiente de reflección')
    # ax1.plot(np.rad2deg(theta_values),np.abs(reflejado_values))
    # ax1.set_ylabel('Módulo de reflectividad')
    
    coeficientes=[reflejado_values,transmitido_values]
    
    return Ex_XZ,Ey_XZ,Ez_XZ,Ex_XY,Ey_XY,Ez_XY,coeficientes

