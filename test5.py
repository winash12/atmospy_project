import numpy as np
from netCDF4 import Dataset,num2date
import sys
import time
file = 'uwnd_1000_2021_26_8_00Z.nc'
nc_uwndFile = Dataset(file,'r')
s = nc_uwndFile.variables['uwnd'][:]
lats = nc_uwndFile.variables['lat'][:]  
lons = nc_uwndFile.variables['lon'][:]
lat = lats[:].squeeze()
lon = lons[:].squeeze()
s = s[0,0,:,:]
lonLen = len(lon)
latLen = len(lat)
dsdy_ref = np.zeros((latLen,lonLen))
dsdy = np.zeros((latLen,lonLen))

rearth = 6371221.3
dj = abs(np.radians((lat[0]-lat[1])) * rearth)



for i in range(0,lonLen):
    s0 = s[0,i]
    s1 = s[1,i]
    if (s0 -999.99 and s1 > -999.99):
        # Make sure derivative is  dq/dy = [q(north) - q(south)]/dlat
        dsdy_ref[0,i] = (s[0,i] - s[1,i])/dj
    else:
        dsdy_ref[0,i] = -999.99
for i in range(0,lonLen):
    if (s[-1,i] > -999.99 and s[-2,i] > -999.99):
        # Make sure derivative is  dq/dy = [q(north) - q(south)]/dlat
        dsdy_ref[-1,i] = (s[-2,i] - s[-1,i])/dj
    else:
        dsdy_ref[-1,i] = -999.99


for j in range(1,latLen-1):
    for i in range(0,lonLen):
        if (s[j-1,i] > -999.99 and s[j+1,i] > -999.99):
            # Make sure derivative is  dq/dy = [q(north) - q(south)]/dlat
            dsdy_ref[j,i] = (s[j+1,i] - s[j-1,i])/(2.0*dj)
        elif (s[j-1,i] < -999.99 and s[j+1,i] > -999.99 and s[j,i] > -999.99):
            dsdy_ref[j,i] = (s[j+1,i] - s[j,i])/dj
        elif (s[j-1,i] > -999.99 and s[j+1,i] < -999.99 and s[j,i] > -999.99):
            dsdy_ref[j,i] = (s[j,i] - s[j-1,i])/dj
        else :
            dsdy_ref[j,i] = -999.99


# North Pole
        
hasNValue = s[0,:] > -999.99
hasNLeft = s[1,:] > -999.99

dsdy[0,:] = -999.99
dsdy[0,:] = np.where(hasNValue & hasNLeft, (s[0,:]-s[1,:])/dj,dsdy[0,:])
#South Pole
hasSRValue = s[-1,:] > -999.99
hasSR2Value = s[-2,:] > -999.99

dsdy[-1,:] = -999.99
dsdy[-1,:] = np.where(hasSRValue & hasSR2Value,(s[-2,:]-s[-1,:])/dj,dsdy[-1,:])


#Regular coordinates
has_value = s[1:-1, :] > -999.99
has_right = s[2:,:] > -999.99
has_left = s[:-2,:] > -999.99
dsdy[1:-1,:] = -999.99
dsdy[1:-1,:] = np.where(has_left & has_value,(s[2,:] - s[1:-1,:]) / dj, dsdy[1:-1,:])
dsdy[1:-1,:] = np.where(has_right & has_value,(s[1:-1,:] - s[:-2,:]) / dj, dsdy[1:-1,:])
dsdy[1:-1,:] = np.where(has_left & has_right,(s[2:,:] - s[:-2,:])/(2.*dj),dsdy[1:-1,:])


print(np.allclose(dsdy_ref,dsdy,1e-14))
#for j in range(0,73):
#    for i in range(0,144):
        #diff = dsdy_ref[j,i] - dsdy[j,i]
        #print(lat[j],dsdy_ref[j,i],dsdy[j,i])
        #print(diff)
