from typing import Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from matplotlib import pyplot as plt
from plot_functions.plot_polarization_elipses import polarization_elipse
from model.focus_field_calculators.base import FocusFieldCalculator
from plot_functions import PlotParameters

def plot_polarization_elipses_on_ax(ax, xmax, ex_values, ey_values, intensity_values):
    x_pos=np.linspace(-xmax*0.95,xmax*0.95,10)
    y_pos=np.linspace(-xmax*0.95,xmax*0.95,10)
    x_values=np.linspace(-xmax,xmax,np.shape(intensity_values)[0])
    y_values=np.linspace(xmax,-xmax,np.shape(intensity_values)[0])
    AMP=np.abs(xmax/6)
    for x_coor in x_pos:
        for y_coor in y_pos:
            x_index = (np.abs(x_values - x_coor)).argmin()
            y_index = (np.abs(y_values - y_coor)).argmin()
            # if np.abs(Ifield_xy[y_index,x_index]/np.max(Ifield_xy))>0.15 and np.abs(ez_XY[y_index,x_index])**2/np.max(Ifield_xy)<0.9:#added a condition for the z intensity to avoid point where the polarization is mostly along z
            if np.abs(intensity_values[y_index,x_index]/np.max(intensity_values))>0.15: #removed the z intensity condition
                if x_coor==x_pos[5] and y_coor==y_pos[5]: #at x,y=0,0 usually the atan2 does not work well
                    x_index+=1
                    y_index+=1
                polarization_elipse(ax,x_coor,y_coor,ex_values[y_index,x_index],ey_values[y_index,x_index],AMP)

def line_plot_on_ax(ax: Axes, title: str, values: list[list], extent: list, horizontal_label: str, vertical_label: str, pad: int = 20):
    ax.set_title(title,pad=pad)
    ax.plot(extent,values)
    ax.grid(True)
    ax.set_xlabel(horizontal_label)
    ax.set_ylabel(vertical_label) 

def color_plot_on_ax(fig: Figure, ax: Axes, title: str, values: list[list], extent: Tuple[int, int], horizontal_label: str, vertical_label: str, colorbar_label: str, square_axis: bool, pad: int = 20, alpha=1, colorbar_ticks: Optional[list] = None):
    pos=ax.imshow(values,extent=extent, interpolation='none', aspect='equal', alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel(horizontal_label)
    ax.set_ylabel(vertical_label)
    if colorbar_ticks:
        cbar= fig.colorbar(pos, ax=ax, ticks=colorbar_ticks)
    else:
        cbar= fig.colorbar(pos, ax=ax)
    cbar.ax.set_ylabel(colorbar_label)
    if square_axis:
        ax.axis('square')

def plot_intensity_at_focus(focus_field: FocusFieldCalculator.FieldAtFocus, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters, params: PlotParameters) -> Figure:
    plt.rcParams['font.size']=14

    #For pasage from (mW/cm^2) to (kW/cm^2) the intensity will be divided by 10**6
    focus_field.calculate_intensity()
    focus_field.Intensity_XY /= 10**6
    focus_field.Intensity_XZ /= 10**6
    focus_field.calculate_intensity_along_x()
    
    fig = plt.figure(num=params.name,figsize=params.size)
    spec = fig.add_gridspec(ncols=3, nrows=2)
    
    radial_pixel_width=focus_field_parameters.x_range*2**0.5/2/np.shape(focus_field.Ex_XZ)[1] #value used to show the pixels centered at the radial position at which they are calculated
    zmax=focus_field_parameters.z_range/2
    rmax=focus_field_parameters.x_range*2**0.5/2 #the maximum radial distance is calculated sqrt(2) times biger than the maximum x or y distnace in the previous functions
    xmax=focus_field_parameters.x_range/2
    extent_XZ = [-rmax-radial_pixel_width,rmax-radial_pixel_width,-zmax,zmax]
    extent_XY = [-xmax-radial_pixel_width,xmax-radial_pixel_width,-xmax+radial_pixel_width,xmax+radial_pixel_width]
    extent_X_axis = np.linspace(-xmax-radial_pixel_width,xmax-radial_pixel_width,np.shape(focus_field.Intensity_XY)[0])
    
    color_plot_on_ax(fig, fig.add_subplot(spec[0, 0]), 'Intensity on the XZ plane', focus_field.Intensity_XZ, extent_XZ, 'x (nm)', 'z (nm)', 'Intensity (kW/cm\u00b2)', square_axis=False)
    color_plot_on_ax(fig, fig.add_subplot(spec[0, 1]), 'Intensity on the XY plane', focus_field.Intensity_XY, extent_XY, 'x (nm)', 'y (nm)', 'Intensity (kW/cm\u00b2)', square_axis=True)
    
    line_plot_on_ax(fig.add_subplot(spec[0, 2]), 'Intensity along x', focus_field.Intensity_along_x, extent_X_axis, 'x (nm)', 'Intensity (kW/cm\u00b2)')
    
    ax = fig.add_subplot(spec[1, 1])
    color_plot_on_ax(fig, ax, 'Polarization on the XY plane', focus_field.Intensity_XY, extent_XY, 'x (nm)', 'y (nm)', 'Intensity (kW/cm\u00b2)', square_axis=True, alpha=0.5)
    plot_polarization_elipses_on_ax(ax, xmax=xmax, ex_values=focus_field.Ex_XY, ey_values=focus_field.Ey_XY, intensity_values=focus_field.Intensity_XY)
    
    '''
    #formula to calculate fwhm
    fwhm=np.abs(2*x_values[np.where(np.abs(Ifield_axis-np.max(Ifield_axis)/2)<0.05*np.max(Ifield_xy))[0]])
    print(fwhm)
    '''
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    
    return fig

def plot_amplitude_and_phase_at_focus(focus_field: FocusFieldCalculator.FieldAtFocus, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters, params: PlotParameters) -> Figure:
    
    xmax=focus_field_parameters.x_range/2
    
    radial_pixel_width=focus_field_parameters.x_range*2**0.5/2/np.shape(focus_field.Ex_XZ)[1]#value used to show the pixels centered at the radial position at which they are calculated
    extent_XY = [-xmax-radial_pixel_width,xmax-radial_pixel_width,-xmax+radial_pixel_width,xmax+radial_pixel_width]
    
    plt.rcParams['font.size']=14

    
    #Amplitude and fase plot
    Amp_max=np.abs(np.max([np.max(np.abs(focus_field.Ex_XY)),np.max(np.abs(focus_field.Ey_XY)),np.max(np.abs(focus_field.Ez_XY))]))**2
    angles_ticks = [5, 90,180,270,355]
    
    fig, ((ax_x1,ax_y1,ax_z1),(ax_x2,ax_y2,ax_z2)) = plt.subplots(num='hola',figsize=params.size,nrows=2, ncols=3)
    
    #Ex
    color_plot_on_ax(fig, ax_x1, '$|E_{f_x}|^2$', np.abs(focus_field.Ex_XY)**2/Amp_max, extent_XY, 'x (nm)', 'y (nm)', 'Relative intensity', True)
    color_plot_on_ax(fig, ax_x2, '$E_{f_x}$ phase', np.angle(focus_field.Ex_XY, deg=True)+180, extent_XY, 'x (nm)', 'y (nm)', 'Phase (degrees)', True, colorbar_ticks=angles_ticks)
    
    #Ey
    color_plot_on_ax(fig, ax_y1, '$|E_{f_y}|^2$', np.abs(focus_field.Ey_XY)**2/Amp_max, extent_XY, 'x (nm)', 'y (nm)', 'Relative intensity', True)
    color_plot_on_ax(fig, ax_y2, '$E_{f_y}$ phase', np.angle(focus_field.Ey_XY, deg=True)+180, extent_XY, 'x (nm)', 'y (nm)', 'Phase (degrees)', True, colorbar_ticks=angles_ticks)

    #ez
    color_plot_on_ax(fig, ax_z1, '$|E_{f_z}|^2$', np.abs(focus_field.Ez_XY)**2/Amp_max, extent_XY, 'x (nm)', 'y (nm)', 'Relative intensity', True)
    color_plot_on_ax(fig, ax_z2, '$E_{f_z}$ phase', np.angle(focus_field.Ez_XY, deg=True)+180, extent_XY, 'x (nm)', 'y (nm)', 'Phase (degrees)', True, colorbar_ticks=angles_ticks)
    
    return fig

