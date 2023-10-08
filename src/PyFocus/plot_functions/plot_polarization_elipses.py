# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:13:16 2020

@author: ferchi
"""

from matplotlib.axis import Axis
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def polarization_elipse(ax: Axis,x_center,y_center,Ex,Ey,Amp):
    beta=np.angle(Ey)-np.angle(Ex)
    theta=np.arctan2(np.abs(Ey),np.abs(Ex))
    inclination=np.arctan(np.tan(2*theta)*np.cos(beta))/2
    A=Amp*((1+(1-np.sin(2*theta)**2*np.sin(beta)**2)**0.5)/2)**0.5
    B=Amp*((1-(1-np.sin(2*theta)**2*np.sin(beta)**2)**0.5)/2)**0.5
    if A>0.2*Amp and B>0.2*Amp:
        inclination=np.arctan2(np.abs(Ey),np.abs(Ex))
        #making beta be between -pi and pi
        if beta>np.pi:
            beta-=2*np.pi
        if beta<-np.pi:
            beta+=2*np.pi        
        if np.abs(beta)>np.pi/2:
            inclination=-np.arctan2(np.abs(Ey),np.abs(Ex))
        ax.add_artist(Ellipse(xy=(x_center,y_center),width=A,height=B,angle=inclination*180/np.pi,fill=False))
        xdist=np.cos(inclination)*0.00001
        ydist=np.sin(inclination)*0.00001
        dx=np.sin(inclination)*B/2
        dy=np.cos(inclination)*B/2
        #correcting direction of turn
        if beta>0:
            ax.arrow(x_center+dx,y_center-dy,xdist,ydist,width=0.01,color="k",head_width=Amp*0.20,head_length=Amp*0.30)
            ax.arrow(x_center-dx,y_center+dy,-xdist,-ydist,width=0.01,color="k",head_width=Amp*0.20,head_length=Amp*0.30)
        else:
            ax.arrow(x_center+dx,y_center-dy,-xdist,-ydist,width=0.01,color="k",head_width=Amp*0.20,head_length=Amp*0.30)
            ax.arrow(x_center-dx,y_center+dy,xdist,ydist,width=0.01,color="k",head_width=Amp*0.20,head_length=Amp*0.30)
    else:
        Amp/=2.5
        if np.abs(beta)<np.pi/2: #it should be =0
            ax.arrow(x_center-1/2*Amp*np.cos(theta),y_center-1/2*Amp*np.sin(theta),Amp*np.cos(theta),Amp*np.sin(theta),width=0.005,color="k",head_width=Amp*0.45,head_length=Amp*0.58)
            ax.arrow(x_center+1/2*Amp*np.cos(theta),y_center+1/2*Amp*np.sin(theta),-Amp*np.cos(theta),-Amp*np.sin(theta),width=0.005,color="k",head_width=Amp*0.45,head_length=Amp*0.58)
        else:
            ax.arrow(x_center+1/2*Amp*np.cos(theta),y_center-1/2*Amp*np.sin(theta),-Amp*np.cos(theta),Amp*np.sin(theta),width=0.005,color="k",head_width=Amp*0.45,head_length=Amp*0.58)
            ax.arrow(x_center-1/2*Amp*np.cos(theta),y_center+1/2*Amp*np.sin(theta),Amp*np.cos(theta),-Amp*np.sin(theta),width=0.005,color="k",head_width=Amp*0.45,head_length=Amp*0.58)

    #draw 1 arrow
if __name__ == '__main__':    
    fig4 = plt.figure()
    spec4 = fig4.add_gridspec(ncols=3, nrows=2)
    ax = fig4.add_subplot(spec4[1, 1])
    Amp=125
    # Ex=-838.2190711891333+1j*462.2721120410149
    # Ey=161.96193971577125-1j*1074.3910757685583
    Ex=1
    Ey=np.exp(+1j*4*np.pi/4)
    polarization_elipse(ax,-427,-427,Ex,Ey,1000)
    ax.set_xlim(-1500,1500)
    ax.set_ylim(-1500,1500)
    plt.tight_layout()

