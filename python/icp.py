import numpy as np 
from stl import mesh
from scipy import spatial
import csv
import math

model = mesh.Mesh.from_file('models/tdrs_small.stl')

x = model.x
y = model.y
z = model.z

points = np.empty((3*x.shape[0], x.shape[1]))
points[:,0] = np.concatenate((x[:,0],x[:,1],x[:,2]), axis=0)
points[:,1] = np.concatenate((y[:,0],y[:,1],y[:,2]), axis=0)
points[:,2] = np.concatenate((z[:,0],z[:,1],z[:,2]), axis=0)

scan = np.loadtxt('frames/frame1_noisy.csv',delimiter=',')


centScan = np.sum(scan,axis=0)/scan.shape[0]

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

theta = np.array([10,60,9])*math.pi/180
R1 = eulerAnglesToRotationMatrix(theta)

modelP = np.empty(scan.shape)
scanP = np.empty(scan.shape)
scanP[:,0] = scan[:,0]
scanP[:,1] = scan[:,1]
scanP[:,2] = scan[:,2]

tree = spatial.KDTree(points)
error = 10
Q = np.eye(3)
while error > 0.1:
	scanP[:,0] = scanP[:,0] - centScan[0]
	scanP[:,1] = scanP[:,1] - centScan[1]
	scanP[:,2] = scanP[:,2] - centScan[2]
	d,i = tree.query(scanP)
	centModel = np.sum(modelP[i,:],axis=0)/len(i)
	modelP[:,0] = modelP[i,0] - centModel[0]
	modelP[:,1] = modelP[i,1] - centModel[1]
	modelP[:,2] = modelP[i,2] - centModel[2]


	W = np.zeros((3,3))
	for j in range(len(i)):
		W += np.matmul(modelP[j,:], np.transpose(scanP[j,:]))

	U, _, V = np.linalg.svd(W, full_matrices=True)
	R = np.matmul(U,np.transpose(V))
	t = centModel - np.matmul(R,centScan)
	for j in range(len(i)):
		scanP[j,:] = np.matmul(R,scanP[j,:])
	scanP = scanP - t
	error = np.linalg.norm(centScan - t)
	centScan = np.sum(scanP,axis=0)/len(i)
	Q = np.matmul(Q,R)
	print(error)
