import numpy as np
from netCDF4 import Dataset,num2date
import sys
import time
file = 'uwnd_1000_2021_26_8_00Z.nc'
nc_uwndFile = Dataset(file,'r')
s = nc_uwndFile.variables['uwnd'][:]
lats = nc_uwndFile.variables['lat'][:]  # extract/copy the data
lons = nc_uwndFile.variables['lon'][:]
lat = lats[:].squeeze()
lon = lons[:].squeeze()


s = s[0,0,:,:]
print(s.shape)


rearth = 6371221.3
dsdx_ref = np.zeros((73,144))

di = abs(np.cos(np.radians(0.0))*rearth*(np.radians(2.5)))



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
# GRIB order - S-N(OUTER)
#              W-E(INNER)

has_left = s[:,:-2] > -999.99
has_right = s[:,2:] > -999.99
has_value = s[:,1:-1] > -999.99
dsdx1 = np.zeros((73,144))
dsdx1[:, 1:-1] = -999.99
dsdx1[:, 1:-1] = np.where(has_right & has_value, (s[:,2:] - s[:,1:-1]) / di, dsdx1[:, 1:-1])
dsdx1[:, 1:-1] = np.where(has_left & has_value, (s[:,1:-1] - s[:,:-2]) / di, dsdx1[:, 1:-1])
dsdx1[:, 1:-1] = np.where(has_left & has_right, (s[:,2:] - s[:,:-2]) / (2. * di), dsdx1[:, 1:-1])

print(np.allclose(dsdx1, dsdx_ref,2.6726184945160235e-15))


for j in range(0,73):
    if (abs(lat[j]) >= 90.0):
        dsdx_ref[j,0] = 0.0
        dsdx_ref[j,-1] =0.0
    elif (np.allclose(2*lon[0]-lon[-1],lon[1],1e-3) or np.allclose(2*lon[0]-lon[-1],lon[1] + 360.0,1e-3)):
        if (s[j, 1] > -999.99 and s[j, -1] > -999.99):
            dsdx_ref[j, 0] = (s[j, 1] - s[j,-1]) / (2.*di)
        elif (s[j,1] < -999.99 and s[j,-1] > -999.99 and s[j,0] > -999.99) :
            dsdx_ref[j,0] = (s[j,1] - s[j,-1]) / di
        elif (s[j, 1] > -999.99 and s[j,-1] > -999.99 and s[j,0] > -999.99):
            dsdx_ref[j,0] = (s [j,1] - s[j,0]) /di
        else:
            dsdx_ref[j, 0] = -999.99
            if (s[j,0] > -999.99 and s[j,-2] > -999.99):
                dsdx_ref[j,-1] = (s[j, 0] - s[j,-2]) / (2. * di)
            elif (s[j,0] < -999.99 and s[j,-2] > -999.99 and s[j,-1] > -999.99):
                dsdx_ref[j,-1] = (s[j,-1] - s[j,-2]) / di
            elif (s[j,0] > -999.99 and s[j,-2] < -999.99 and s[j,-1] > -999.99) :
                dsdx_ref[j,-1] = (s[j,1] - s[j,-1]) / di
            else:
                dsdx_ref[j, -1] = -999.99
    elif (np.allclose(lon[0],lon[-1],1e-3)):
        if (s[j, 1] > -999.99 and s[j,-2] > -999.99) :
            dsdx_ref[j,0] = (s[j,1] - s[j,-2]) / (2. * di)
        elif (s[j,1] < -999.99 and s[j,-2] > -999.99 and s[j,0] > -999.99) :
            dsdx_ref[j,0] = (s[j,0] - s[j,-2]) / di
        elif (s[1, j] > -999.99 and s[- 2, j] < -999.99 and s[0, j] > -999.99):
            dsdx_ref[j,0] = (s[j,1] - s[j,0]) / di
        else:
            dsdx_ref[j,0] = -999.99
            dsdx_ref[j,-1] = dsdx_ref[j,0]
    else:
        #print(lat[j])
        if (s[j, 1] > -999.99 and s[j,0] > -999.99) :
            dsdx_ref[j,0] = (s[j,1] - s[j,0]) /di
            #print(lat[j],dsdx_ref[j,0])
            #sys.exit()
        else:
            dsdx_ref[j,0] = -999.99
            
            if (s[j,-1] > -999.99 and s[j,-2] > -999.99):
                dsdx_ref[j,-1] = (s[j,-1] -s[j,-2]) /di
            else:
                dsdx_ref[j,-1] = -999.99

t1 = time.time()
total = t1-t0
print(total)




      
hasValue = s[1:-1,0] > -999.99
hasRight = s[1:-1,-1] > -999.99
hasLeft = s[1:-1,1] > -999.99
hasRight2 = s[1:-1,-2] > -999.99



if (np.allclose(2*lon[0]-lon[-1],lon[1],1e-3) or np.allclose(2*lon[0]-lon[-1],lon[1] + 360.0,1e-3)):
    dsdx1[1:-1,0] = -999.99
    dsdx1[1:-1,-1] = -999.99
    dsdx1[1:-1,0] = np.where(hasRight & hasValue,(s[1:-1,-1] - s[1:-1,0]) / di, dsdx1[1:-1, 0])
    dsdx1[1:-1,0] = np.where(hasLeft & hasValue,(s[1:-1,1] - s[1:-1,0]) / di, dsdx1[1:-1, 0])
    dsdx1[1:-1,0] = np.where(hasLeft & hasRight,(s[1:-1,1] - s[1:-1,-1]) /2. * di, dsdx1[1:-1, 0])
    dsdx1[1:-1,-1] = np.where(hasRight & hasRight2,(s[1:-1,-1] - s[1:-1,-2]) / di, dsdx1[1:-1, -1])
    dsdx1[1:-1,-1] = np.where(hasLeft & hasRight,(s[1:-1,1] - s[1:-1,-1]) / di, dsdx1[1:-1, -1])
    dsdx1[1:-1,-1] = np.where(hasValue & hasRight2,(s[1:-1,0] - s[1:-1,-2]) /2. * di, dsdx1[1:-1, -1])
elif (np.allclose(lon[0],lon[-1],1e-3)):
    dsdx1[1:-1,0] = -999.99
    dsdx1[1:-1,-1] = -999.99
    dsdx1[1:-1,0] = np.where(hasLeft & hasRight2,(s[1:-1,1] - s[1:-1,-2]) / 2. *di, dsdx1[1:-1, 0])
    dsdx1[1:-1,0] = np.where(hasValue & hasRight2,(s[1:-1,0] - s[1:-1,-2]) /di, dsdx1[1:-1, 0])
    dsdx1[1:-1,0] = np.where(hasLeft & hasValue,(s[1:-1,1] - s[1:-1,0]) / di, dsdx1[1:-1, 0])
else:
    dsdx1[1:-1,0] = -999.99
    dsdx1[1:-1,-1] = -999.99
    dsdx1[1:-1,0] = np.where(hasLeft & hasValue,(s[1:-1,1] - s[1:-1,0]) / di, dsdx1[1:-1, 0])
    dsdx1[1:-1,-1] = np.where(hasRight & hasRight2,(s[1:-1,-1] - s[1:-1,-2]) / di, dsdx1[1:-1, -1])

sys.exit()

