import numpy as np
import warnings

from netCDF4 import Dataset
import scipy.ndimage as ndimage
from windspharm.standard import VectorWind
from itertools import islice
import numpy.random as rnd
import sys
def main():

    file = 'uwnd_925_2021_9_9_00Z.nc'
    file1 = 'vwnd_925_2021_9_9_00Z.nc'
    nc_uwndFile = Dataset(file,'r')
    nc_vwndFile = Dataset(file1,'r')
    s = nc_uwndFile.variables['uwnd'][:]
    s1 = nc_vwndFile.variables['vwnd'][:]
    lats = nc_uwndFile.variables['lat'][:]  
    lons = nc_uwndFile.variables['lon'][:]
    thta = [260.,270.,280.,290.,300.,310.,320.,330.,340.,350.,360.,370.,380.,390.,400.,410.,420.] 
    thta = np.asarray(thta,dtype=np.float64)

    sipv(lats,lons,thta)
    
def sipv(lats,lons,thta):
    
    latLen = 73
    lonLen = 144
    kthta = 17

    uthta = rnd.randn(17,73, 144)
    uthta[rnd.randn(17,73, 144) < 0.1] = -999.99

    vthta = rnd.randn(17,73, 144)
    vthta[rnd.randn(17,73, 144) < 0.1] = -999.99

    pthta = rnd.randn(17,73, 144)
    #pthta[rnd.randn(17,73, 144) < 0.1] = -999.99
    
    ipv = np.empty((kthta,latLen,lonLen))
    p0 = 100000.
    kappa = 2./7.
    gravity = 9.80665
    rearth = 6371221.3
    np.seterr(all='warn')
    warnings.filterwarnings('error')


    k = 0
    absVor = np.empty((17,73,144))        
    for uthta2d,vthta2d in zip(uthta,vthta):

        w = VectorWind(uthta2d,vthta2d)
        absVor[k,:,:]=w.absolutevorticity()
        k += 1
    hasNoPthta = pthta[1:-1,:,:] < 0.
    hasNoPthta0 = pthta[0:-2,:,:] < 0.
    hasNoAbsVor = absVor[1:-1,:,:] < -999.99

    ipv[1:-1,:,:] = np.where(hasNoPthta|hasNoPthta0|hasNoAbsVor,-999.99,ipv[1:-1,:,:])

    tdwn = thta[0:-2,None,None]*(pthta[0:-2,:,:]/p0)
    tup =  thta[1:-1,None,None] * (pthta[1:-1,:,:]/p0)
    dlt = np.log(tup/tdwn)
    dlp = np.log(pthta[1:-1,:,:]/pthta[0:-2,:,:])
    dltdlp = dlt/dlp
    stabl = (thta[k]/pthta[1:-1,:,:]) *(dltdlp-kappa)
    ipv[1:-1,:,:] = -gravity * absv[1:-1,:,:] * stabl
    ipv[1:-1,:,:] = ndimage.gaussian_filter(ipv[1:-1,:,:]*1e6,sigma=2,order=0)
    return ipv
main()
