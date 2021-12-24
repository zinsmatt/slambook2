import numpy as np
from scipy.spatial.transform.rotation import Rotation as Rot



p = np.random.uniform(-10, 10, 3)

rot = Rot.from_euler("xyz", np.random.uniform(-40, 40, 3), degrees=True)
q = rot.as_quat()


# q = (x, y, z, w)
p2 = rot.as_matrix().dot(p)

uv = 2*np.cross(q[:3], (p))
p22 = p + q[3] * uv + np.cross(q[:3], (uv))


print(p2)
print(p22)

from sympy import Symbol, Matrix, diff

x = Symbol('x3')
y = Symbol('y3')
z = Symbol('z3')
qx = Symbol('qx')
qy = Symbol('qy')
qz = Symbol('qz')
qw = Symbol('qw')


q = Matrix([qx, qy, qz])
p = Matrix([x, y, z])

uv = 2 * q.cross(p)

Xc = p + qw * uv + q.cross(uv)
print(diff(Xc[0], 'qx'))
print(diff(Xc[0], 'qy'))
print(diff(Xc[0], 'qz'))
print(diff(Xc[0], 'qw'))
print()
print(diff(Xc[1], 'qx'))
print(diff(Xc[1], 'qy'))
print(diff(Xc[1], 'qz'))
print(diff(Xc[1], 'qw'))
print()
print(diff(Xc[2], 'qx'))
print(diff(Xc[2], 'qy'))
print(diff(Xc[2], 'qz'))
print(diff(Xc[2], 'qw'))
print()