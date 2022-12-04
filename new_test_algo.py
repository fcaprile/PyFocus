from pydantic import BaseModel

class A(BaseModel):
    a: int
    
    @property
    def hola(self):
        return f'hola {self.a}'

a = A(a=1)
print(a)
print(a.hola)
