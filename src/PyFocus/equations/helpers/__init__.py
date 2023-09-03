import numpy as np

def cart2pol(x,y, round_r: bool = False):    
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)
    if round_r: r = int(np.rint(r))
    return t,r
