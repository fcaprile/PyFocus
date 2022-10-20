from matplotlib import pyplot as plt
import numpy as np

def plot_objective_field(Ifield, Ex, Ey, xmax, figure_name='', font_size=20, folder=''):
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
