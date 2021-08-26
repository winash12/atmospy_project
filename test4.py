import numpy as np
from netCDF4 import Dataset,num2date
import sys
import time
file = 'uwnd_1000_2021_23_8_00Z.nc'
nc_uwndFile = Dataset(file,'r')
s = nc_uwndFile.variables['uwnd'][:]
s = s[0,0,:,:]
print(s.shape)


rearth = 6371221.3
dsdx_ref = np.zeros((73,144))

di = abs(np.cos(np.radians(0.0))*rearth*(np.radians(2.5)))


t0 = time.time()
for j in range(0,73):                
    for i in range(1, 144-1):
        if (s[j, i+1] > -999.99 and s[j,i-1] > -999.99):
            dsdx_ref[j, i] = (s[j,i+1] - s[j,i-1])/(2.*di)
        elif (s[j,i+1] <= -999.99 and s[j,i-1] > -999.99 and s[j,i] > -999.99):
            dsdx_ref[j,i] = (s[j,i] - s[j,i-1])/di
        elif (s[j,i+1] > -999.99 and s[j,i-1] <= -999.99 and s[j,i] > -999.99):
            dsdx_ref[j,i]  = (s[j,i+1] - s[j,i])/di
        else:
            dsdx_ref[j,i] = -999.99
t1 = time.time()
total = t1-t0
print(total)
            
# GRIB order - S-N(OUTER)
#              W-E(INNER)
t2 = time.time()
has_left = s[:,:-2] > -999.99
has_right = s[:,2:] > -999.99
has_value = s[:,1:-1] > -999.99
dsdx1 = np.zeros((73,144))
dsdx1[:, 1:-1] = -999.99
dsdx1[:, 1:-1] = np.where(has_right & has_value, (s[:,2:] - s[:,1:-1]) / di, dsdx1[:, 1:-1])
dsdx1[:, 1:-1] = np.where(has_left & has_value, (s[:,1:-1] - s[:,:-2]) / di, dsdx1[:, 1:-1])
dsdx1[:, 1:-1] = np.where(has_left & has_right, (s[:,2:] - s[:,:-2]) / (2. * di), dsdx1[:, 1:-1])
t3 = time.time()

print(t3-t2)

print(np.allclose(dsdx1, dsdx_ref,2.6726184945160235e-15))

hasValue = s[:,0] > -999.99
hasRight = s[:,-1] > -999.99
hasLeft = s[:,1] > -999.99
hasRight2 = s[:,-2] > -999.99


if (np.allclose(2*lon[0]-lon[-1],lon[1],1e-3) or np.allclose(2*lon[0]-lon[-1],lon[1] + 360.0,1e-3)):
    dsdx1[:,0] = -999.99
    dsdx1[:,-1] = -999.99
    dsdx1[:,0] = np.where(hasRight & hasValue,(s[:,-1] - s[:,0) / di, dsdx1[:, 0]))
    dsdx1[:,0] = np.where(hasLeft & hasValue,(s[:,1] - s[:,0) / di, dsdx1[:, 0]))
    dsdx1[:,0] = np.where(hasLeft & hasRight,(s[:,1] - s[:,-1) /2. * di, dsdx1[:, 0]))
    dsdx1[:,-1] = np.where(hasRight & hasRight2,(s[:,-1] - s[:,-2) / di, dsdx1[:, -1]))
    dsdx1[:,-1] = np.where(hasLeft & hasRight,(s[:,1] - s[:,-1) / di, dsdx1[:, -1]))
    dsdx1[:,-1] = np.where(hasValue & hasRight2,(s[:,0] - s[:,-2) /2. * di, dsdx1[:, -1]))
elif (np.allclose(lon[0],lon[-1],1e-3)):

    dsdx1[:,0] = -999.99
    dsdx1[:,-1] = -999.99
    dsdx[:,0] = np.where(hasLeft & hasRight2,(s[:,1] - s[:,-2) / 2. *di, dsdx1[:, 0]))
    dsdx[:,0] = np.where(hasValue & hasRight2,(s[:,0] - s[:,-2) /di, dsdx1[:, 0]))
    dsdx[:,0] = np.where(hasLeft & hasValue,(s[:,1] - s[:,0) / di, dsdx1[:, 0]))
else:
    dsdx[:,0] = -999.99
    dsdx[j,-1] = -999.99
    dsdx[:,0] = np.where(hasLeft & hasValue,(s[:,1] - s[:,0) / di, dsdx1[:, 0]))
    dsdx[j,-1] = np.where(hasRight & hasRight2,(s[:,-1] - s[:,-2) / di, dsdx1[:, -1]))



