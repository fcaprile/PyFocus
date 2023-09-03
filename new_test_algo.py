from dataclasses import dataclass
import numpy as np

@dataclass
class InterfaceParameters:
    axial_position: int # Axial location of the interface
    ns: np.ndarray # Array of refraction indexes
    ds: np.ndarray # Array of thinckness of each layer (nm)
    
print(InterfaceParameters(1, np.array((1,1)),  np.array((2,2))))