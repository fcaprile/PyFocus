from dataclasses import dataclass
from enum import Enum, auto

@dataclass
class CustomMaskParameters:
    divisions_theta: int = 100
    divisions_phi: int = 100
    
class PlotPlanes(Enum):
    XZ = auto()
    XY = auto()
    YZ = auto()