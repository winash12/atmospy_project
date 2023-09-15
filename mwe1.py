import os.path
import xarray as xr
import numpy as np
import boto3
import metpy.calc as mpcalc
from botocore import UNSIGNED
from botocore.config import Config
import sys

if (not os.path.isfile('gfs.t12z.pgrb2.0p50.f000')):

    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    client.download_file('noaa-gfs-bdp-pds', 'gfs.20230824/12/atmos/gfs.t12z.pgrb2.0p25.f000', 'gfs.t12z.pgrb2.0p25.f000')

u850 = xr.open_dataset('gfs.t12z.pgrb2.0p50.f000', engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'u', 'level': 850}})
u = u850.u
v850 = xr.open_dataset('gfs.t12z.pgrb2.0p50.f000', engine='cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'v', 'level': 850}})
v = v850.v

# Compute the 850 hPa relative vorticity.
vort850 = mpcalc.vorticity(u, v)


# Compute the 850 hPa divergence.
div850 = mpcalc.divergence(u, v)



mask = ((vort850.latitude <= 13.5) & (vort850.latitude >= 5.0) & (vort850.longitude <= 202.) & (vort850.longitude >= 191.))
vortmask = vort850.where(mask)

vortmask = vortmask.fillna(0.0)
divmask = div850.where(mask)
divmas = div850.fillna(0.0)
dx, dy = mpcalc.lat_lon_grid_deltas(vortmask.longitude, vortmask.latitude)


upsi = xr.zeros_like(vortmask)
print(upsi.shape)


vpsi = xr.zeros_like(vortmask)
print(vpsi.shape)
uchi = xr.zeros_like(divmask)
vchi = xr.zeros_like(divmask)
x_ll = list(vortmask.longitude.values).index(191.0)
x_ur = list(vortmask.longitude.values).index(202.0)
y_ll = list(vortmask.latitude.values).index(5.0)
y_ur = list(vortmask.latitude.values).index(13.5)

x_ll_subset = list(vortmask.longitude.values).index(180.0)
x_ur_subset = list(vortmask.longitude.values).index(220.0)
y_ll_subset = list(vortmask.latitude.values).index(0.0)
y_ur_subset = list(vortmask.latitude.values).index(30.0)

i = np.abs(x_ll_subset-x_ur_subset)
j = np.abs(y_ll_subset-y_ur_subset)

istart = np.linspace(x_ll_subset,x_ur_subset,num = i,endpoint=False,dtype=np.int32)


jstart = np.linspace(y_ur_subset,y_ll_subset,num=j,endpoint=False,dtype=np.int32)

x = np.abs(x_ll-x_ur)
y = np.abs(y_ll-y_ur)


xstart = np.linspace(x_ll,x_ur,num = x,endpoint=False,dtype=np.int32)
ystart = np.linspace(y_ll,y_ur,num=y,endpoint=False,dtype=np.int32)

xindex,yindex = np.meshgrid(xstart,ystart)

iindex = np.zeros((x,y))
jindex = np.zeros((x,y))

for i in range(x_ll_subset, x_ur_subset):

    for j in range(y_ur_subset, y_ll_subset): 

        iindex[:,:] = i
        jindex[:,:] = j
        xdiff = (iindex[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset]-xindex[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset])*dx[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset]
        ydiff = (jindex[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset]-yindex[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset])*dy[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset]
        rsq = (xdiff*xdiff) + (ydiff*ydiff)
        upsi[j,i] = np.where(rsq > 0, vortmask[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset]*-1.0*(ydiff/rsq)*dx[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset]*dy[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset], 0.0).sum()
        vpsi[j,i] = np.where(rsq > 0, vortmask[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset]*-1.0*(xdiff/rsq)*dx[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset]*dy[y_ur_subset:y_ll_subset,x_ll_subset:x_ur_subset], 0.0).sum()
        
                #   xdiff = (i-x1)*dx[y1,x1].magnitude
                 #   ydiff = (j-y1)*dy[y1,x1].magnitude
                  #  rsq = (xdiff*xdiff) + (ydiff*ydiff)
                # Compute the non-divergent flow contribution.
                    #upsi[j,i] += vortmask[y1,x1].values * -1.0 * (ydiff / rsq) * dx[y1,x1].magnitude * dy[y1,x1].magnitude
                    #vpsi[j,i] += vortmask[y1,x1].values * (xdiff / rsq) * dx[y1,x1].magnitude * dy[y1,x1].magnitude
                    #print(upsi[j,i].values)





upsi[:,:] = (1/(2*np.pi)) * upsi[:,:]
vpsi[:,:] = (1/(2*np.pi)) * vpsi[:,:]
