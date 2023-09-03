from pydantic import BaseModel, StrictBool, StrictStr

class PlotParameters(BaseModel):
    name: StrictStr
    size: tuple = (16, 8)
