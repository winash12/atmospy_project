import numpy as np
from netCDF4 import Dataset,num2date
import numpy.random as rnd
import sys
import time
import math
rearth = 6371221.3
file = 'uwnd_925_2021_9_9_00Z.nc'
file1 = 'vwnd_925_2021_9_9_00Z.nc'
nc_uwndFile = Dataset(file,'r')
nc_vwndFile = Dataset(file1,'r')
s = nc_uwndFile.variables['uwnd'][:]
s1 = nc_vwndFile.variables['vwnd'][:]
lats = nc_uwndFile.variables['lat'][:]  
lons = nc_uwndFile.variables['lon'][:]
print(lats.shape,lons.shape)

lat = lats[:].squeeze()
lon = lons[:].squeeze()

dvdx = rnd.randn(73, 144)
dvdx[rnd.randn(73, 144) < 0.1] = -999.99

dudy = rnd.randn(73, 144)
dudy[rnd.randn(73, 144) < 0.1] = -999.99

u = s[0,0,:,:]
lonLen = len(lon)
latLen = len(lat)
relv_ref = np.zeros((73,144))
relv = np.zeros((73,144))
missing = 0
# Begin South Pole
for i in range(0,lonLen):
    if (u[1,i] > -999.99):
        relv_ref[0,0] = u[1,i]
        break
    else:
        missing = i

for i in range(missing+2,lonLen):
    if (u[1,i] > -999.99):
        relv_ref[0,0] = relv_ref[0,0]+u[1,i]
    else:
        missing = missing +1
if (abs(lon[0] - lon[-1]) > .0001) :
    if (u[-1,1] > -999.99):
        relv_ref[0,0]= relv_ref[0,0]+u[-1,1]
    else:
        missing = missing +1
    relv_ref[0,0] = relv_ref[0,0]* math.cos(math.radians(lat[1]))/(1.-math.sin (math.radians(lat[1])))/rearth /float(lonLen - missing)
else:
    relv_ref[0,0] = relv_ref[0,0]* math.cos(math.radians(lat[1]))/(1.-math.sin (math.radians(lat[1])))/rearth /float(lonLen -1- missing)
for i in range(1,lonLen):
    relv_ref[0,i] = relv_ref[0,0]


#Begin  North Pole
missing = 0
for i in range(0,lonLen):
    if (u[-2,i] > -999.99):
        relv_ref[-1,-1] = u[-2,i]
        break
    else:
        missing = i
for i in range(missing+2,lonLen):
    if (u[-2,i] > -999.99) :
        relv_ref[-1,-1] = relv_ref[-1,-1]+u[-2,i]
    else:
        missing = missing +1
if (abs(lon[0] - lon[-1] > .001)):
    if(u[-2,i] > -999.99):
        relv_ref[-1,-1]=relv_ref[-1,-1]+u[-2,-1]
    else:
        missing = missing + 1
    relv_ref[-1,-1]=relv[-1,-1]* np.cos(np.radians(lat[-2]))/(1.-np.sin(np.radians(lat[-2])))/rearth /float(lonLen -1- missing)
else:
    relv_ref[-1,-1]=relv_ref[-1,-1]* np.cos(math.radians(lat[-2]))/(1.-np.sin (np.radians(lat[-2])))/rearth /float(lonLen - 2-missing)        
for i in range(0,lonLen-1):
    relv_ref[-1,i] = relv_ref[-1,-1]

          

uSouth = u[1,:]
uSouthMissing = (u[1,:] < -999.99).sum()
relv[0,:] = uSouth[uSouth > -999.99].sum()
sphericalCapFactor = math.cos(math.radians(lat[1]))/(1.-math.sin (math.radians(lat[1])))/rearth /float(lonLen-uSouthMissing)
relv[0,:] = relv[0,:]*sphericalCapFactor

uNorth = u[-2,:]
uNorthMissing = (u[-2,:] < -999.99).sum()
relv[-1,:] = uNorth[uNorth > -999.99].sum()
sphericalCapFactor = math.cos(math.radians(lat[-2]))/(1.-math.sin (math.radians(lat[-2])))/rearth /float(lonLen-uNorthMissing)
relv[-1,:] = relv[-1,:]*sphericalCapFactor


#for j in range(0,144):
    #print(relv[0,j],relv_ref[0,j])
    #print(relv[-1,j],relv_ref[-1,j])


## Global
for j in range(1,latLen-1):
    for i in range(0,lonLen):
        if (u[j,i] < -999.99 or dvdx[j,i] < -999.99 or dudy[j,i] < -999.99 ):
            relv_ref[j,i] = -999.99
        else:
            relv_ref[j,i] = dvdx[j,i]-dudy[j,i]+(u[j,i]*np.tan(np.radians(lat[j])))/rearth


hasNoU = u[1:-1,:] < -999.99
hasNoDvDx = dvdx[1:-1,:] < -999.99
hasNoDuDy = dudy[1:-1,:] < -999.99


relv[1:-1,:] = np.where(hasNoU| hasNoDvDx |hasNoDuDy,-999.99,relv[1:-1,:])


relv[1:-1,:] = dvdx[1:-1,:]-dudy[1:-1,:]+(u[1:-1,:]*np.tan(np.radians(lat[1:-1,None])))/rearth

#for j in range(1,72):
#    for i in range(0,144):
        #print(relv[j,i],relv_ref[j,i])


absv_ref = np.empty((latLen,lonLen))
omega = 7.29212e-05
for j in range(0,latLen):
    corl = 2.0 * omega * np.sin(np.radians(lat[j]))
    for i in range(0,lonLen):
        if (relv_ref[j,i] > -999.99):
            absv_ref[j,i] = relv_ref[j,i]+corl
        else:
            absv_ref[j,i] = -999.99

corl = np.empty((latLen))
corl = 2.0 * omega * np.sin(np.radians(lat))
corl = corl[:,None]
hasNoRelv = relv[:,:] < -999.99
absv = np.empty((latLen,lonLen))
absv = np.where(hasNoRelv,-999.99,absv[:,:])
absv  = relv + corl

for j in range(0,73):
    for i in range(0,144):
        print(absv[j,i],absv_ref[j,i])
