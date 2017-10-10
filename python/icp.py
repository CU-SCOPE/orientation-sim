from kdtree import kdTree
from stl import mesh
import math
import numpy as np

model = mesh.Mesh.from_file('models/galileo_red.stl')
model.rotate([-0.5, 0.0, 0.0], math.radians(90))
x = model.x
y = model.y
z = model.z

tree = kdTree(x,y,z)
points = tree.points
scan = np.loadtxt('frames/frame1.csv',delimiter=',')
scan = scan + [0,0,4]
d,cp = tree.query(scan)
print(d)