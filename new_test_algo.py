'''
import numpy as np

x = np.array([[0],[1],[2],[3]])
y = np.array((0,1,2,3))

bro = np.broadcast(x, y)
result = np.empty(bro.shape)
result.flat = [u**2+v**2 for (u,v) in bro]
'''
'''
result = np.zeros((4,4))

for i, a in enumerate(a_values):
    for j, b in enumerate(b_values):
        result[i,j] = (a**2+b**2)**0.5
'''
'''
print(result)
'''

a=[-500.      , -485.71428571, -471.42857143, -457.14285714, -442.85714286,
 -428.57142857, -414.28571429, -400.        , -385.71428571, -371.42857143,
 -357.14285714, -342.85714286, -328.57142857, -314.28571429, -300.,
 -285.71428571, -271.42857143, -257.14285714, -242.85714286, -228.57142857,
 -214.28571429, -200.        , -185.71428571, -171.42857143, -157.14285714,
 -142.85714286, -128.57142857, -114.28571429, -100.        ,  -85.71428571,
  -71.42857143,  -57.14285714,  -42.85714286,  -28.57142857,  -14.28571429,
    0.        ,   14.28571429,   28.57142857,   42.85714286,   57.14285714,
   71.42857143,   85.71428571,  100.        ,  114.28571429,  128.57142857,
  142.85714286,  157.14285714,  171.42857143,  185.71428571,  200.,
  214.28571429,  228.57142857,  242.85714286,  257.14285714,  271.42857143,
  285.71428571,  300.        ,  314.28571429,  328.57142857,  342.85714286,
  357.14285714,  371.42857143,  385.71428571,  400.        ,  414.28571429,
  428.57142857,  442.85714286,  457.14285714,  471.42857143,  485.71428571,
  500.        ]