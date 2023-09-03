from enum import Enum, auto


class MaskType(Enum):
    no_mask = auto()
    vortex_mask = auto()
    propagated_vortex_mask = auto()
    custom_mask = auto()
