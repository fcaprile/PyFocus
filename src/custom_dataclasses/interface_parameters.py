import pydantic
import numpy as np

class Config:
    arbitrary_types_allowed = True

@pydantic.dataclasses.dataclass(config=Config)
class InterfaceParameters:
    axial_position: int # Axial location of the interface
    ns: np.ndarray # Array of refraction indexes
    ds: np.ndarray # Array of thinckness of each layer (nm)