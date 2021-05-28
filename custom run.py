# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:22:57 2021

@author: ferchi
"""
from main import PyFocus

sim=PyFocus()

NA=1.4
n=1.5
h=3
f=h*n/NA
w0=5
wavelength=640
gamma=45
beta=90
zp0=0
rsteps=50
zsteps=80
field_of_view=1000
z_field_of_view=2000
I0=1
parameters=NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0
#VPP mask:
fields=sim.VPP(False,False,*parameters)


sim.plot(*fields,field_of_view,z_field_of_view)
