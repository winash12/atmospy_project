import numpy as np
import os.path
import xarray as xr
import numpy as np
import boto3
import metpy.calc as mpcalc
from botocore import UNSIGNED
from botocore.config import Config
import sys


class vortdiv_inversion:

    def __init__(self,outerBBlat1,outerBBlon1,outerBBlat2,outerBBlon2,innerBBlat1,innerBBlon1,innerBBlat2,innerBBlon2):
        
        self.u850 = None
        self.v850 = None
        self.u = None
        self.v = None
        self.vort850 = None
        self.div850 = None
        self.vortMask = None
        self.divMask = None
        self.dx = None
        self.dy = None
        self.upsi = None
        self.vpsi = None
        self.uchi = None
        self.vchi = None
        
    def readFile(self):
        if (not os.path.isfile('gfs.t12z.pgrb2.0p50.f000')):

            client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
            client.download_file('noaa-gfs-bdp-pds', 'gfs.20230824/12/atmos/gfs.t12z.pgrb2.0p25.f000', 'gfs.t12z.pgrb2.0p25.f000')

            self.u850 = xr.open_dataset('gfs.t12z.pgrb2.0p50.f000', engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'u', 'level': 850}})

            self.v850 = xr.open_dataset('gfs.t12z.pgrb2.0p50.f000', engine='cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'v', 'level': 850}})

        def setData(self):
            self.u = self.u850.u
            self.v = self.v850.v

        def calculateVorticity(self):
            self.vort850 = mpcalc.vorticity(self.u,self.v)

        def calculateDivergenec(self):

            self.div850 = mpcalc.divergence(self.u,self.v)

        def vorticityMask(self):
            
            mask = ((self.vort850.latitude <= 13.5) & (self.vort850.latitude >= 5.0) & (self.vort850.longitude <= 202.) & (self.vort850.longitude >= 191.))
            self.vortmask = vort850.where(mask)
            self.vortmask = vortmask.fillna(0.0)

        def divergenceMask(self):
            mask = ((self.div850.latitude <= 13.5) & (self.div850.latitude >= 5.0) & (self.div850.longitude <= 202.) & (self.div850.longitude >= 191.))
            self.divmask = self.div850.where(mask)
            self.divmask = self.div850.fillna(0.0)

        def calculateDistanceMatrix(self):
            self.dx, self.dy = mpcalc.lat_lon_grid_deltas(self.vortmask.longitude, self.vortmask.latitude)
            self.dx = np.abs(dx)
            self.dy = np.abs(dy)

        def initiailizeRotationalAndIrrotationalWind(self):
            self.upsi = xr.zeros_like(self.vortmask)
            self.vpsi = xr.zeros_like(self.vortmask)
            self.uchi = xr.zeros_like(self.divmask)
            self.vchi = xr.zeros_like(self.divmask)

        def defineBoundingBoxIndices(self):
            x_ll = list(vortmask.longitude.values).index(191.0)
            x_ur = list(vortmask.longitude.values).index(202.0)
            y_ll = list(vortmask.latitude.values).index(5.0)
            y_ur = list(vortmask.latitude.values).index(13.5)
            
            x_ll_subset = list(vortmask.longitude.values).index(180.0)
            x_ur_subset = list(vortmask.longitude.values).index(220.0)
            y_ll_subset = list(vortmask.latitude.values).index(0.0)
            y_ur_subset = list(vortmask.latitude.values).index(30.0)

        def calculateRotationalWindFromInversion(self):
            for i in range(x_ll_subset, x_ur_subset):
                for j in range(y_ur_subset, y_ll_subset): 

                    # Computing the contribution to each other grid point (masked area).
                    # x1 and y1 refer to x' and y' in the Oertel and Schemm (2021) equations.
                    for x1 in range(x_ll, x_ur):
                        for y1 in range(y_ur, y_ll):
                        outer_point = [i,j]
                        inner_point = [x1,y1]
                        if inner_point != outer_point:
                            
                            # Compute x-x', y-y', and r^2...
                            xdiff = (i-x1)*dx[y1,x1].magnitude
                            ydiff = (j-y1)*dy[y1,x1].magnitude
                            rsq = (xdiff*xdiff) + (ydiff*ydiff)
                            
                            # Compute the non-divergent flow contribution.
                            upsi[j,i] += vortmask[y1,x1].values * -1.0 * (ydiff / rsq) * dx[y1,x1].magnitude * dy[y1,x1].magnitude
                            vpsi[j,i] += vortmask[y1,x1].values * (xdiff / rsq) * dx[y1,x1].magnitude * dy[y1,x1].magnitude
                    
            upsi[:,:] = (1/(2*np.pi)) * upsi[:,:]
            vpsi[:,:] = (1/(2*np.pi)) * vpsi[:,:]


            def calculateDivergentWindFromInversion(self):

                for i in range(x_ll_subset, x_ur_subset):
                    for j in range(y_ur_subset, y_ll_subset): 
                        
                        # Computing the contribution to each other grid point (masked area).
                        # x1 and y1 refer to x' and y' in the Oertel and Schemm (2021) equations.
                        for x1 in range(x_ll, x_ur):
                            for y1 in range(y_ur, y_ll):
                                outer_point = [i,j]
                                inner_point = [x1,y1]
                                if inner_point != outer_point:
                                    
                                    # Compute x-x', y-y', and r^2...
                                    xdiff = (i-x1)*dx[y1,x1].magnitude
                                    ydiff = (j-y1)*dy[y1,x1].magnitude
                                    rsq = (xdiff*xdiff) + (ydiff*ydiff)
                                    
                                    # Compute the irrotational flow contribution.
                                    uchi[j,i] += divmask[y1,x1].values * (xdiff / rsq) * dx[y1,x1].magnitude * dy[y1,x1].magnitude
                                    vchi[j,i] += divmask[y1,x1].values * (ydiff / rsq) * dx[y1,x1].magnitude * dy[y1,x1].magnitude
                    

            uchi[:,:] = (1/(2*np.pi)) * uchi[:,:]
            vchi[:,:] = (1/(2*np.pi)) * vchi[:,:]


