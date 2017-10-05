import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

def readscanpcd(filename):
    x=[]
    y=[]
    z=[]
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=' ')
        try:
            # skip headerlines
            for i in range(11):
                reader.next()
                
            # get values for scanner endpoints
            for row in reader:
                x.append(float(row[0]))
                y.append(float(row[1]))
                z.append(float(row[2]))
                
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))

    return np.array(x), np.array(y), np.array(z)

x,y,z = readscanpcd('framerotated_noisy00000.pcd')
x = x[np.isfinite(x)]
y = y[np.isfinite(y)]
z = z[np.isfinite(z)]
arr = np.empty((x.shape[0],3))
arr[:,0] = x
arr[:,1] = y
arr[:,2] = z
np.savetxt('frame1_noisy.csv', arr, delimiter=',')