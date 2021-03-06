#!/usr/bin/python3.5
import sys,math
import warnings
import numpy as np
import warnings
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib as mpl

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import scipy.ndimage as ndimage
from windspharm.standard import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim
from decimal import *

class potential_vorticity:


    def latlon(self,ni,nj,lat1,lat2,lon1,lon2):
        lon = np.empty(ni)
        lat = np.empty(nj)
        lon[0] = lon1
        lon[-1] = lon2
        if (lon1 < 0.0):
            lon[1] = lon[1]  + 360.0
        if (lon2 <= 0.0):
            lon[-1] = lon[-1]+360.0
        for i in range(0, ni):
            lon[i] = lon[0] + float(i)*(lon[-1] - lon[0])/float(ni-1)
            if (lon[i] > 180.0):
                lon[i] = lon[i] - 360.

        lat[0] = lat1
        lat[-1] = lat2

        if (lat2 > 90.0):
            lat[-1] = 180. - lat[-1]
        if (lat2 < -90.0):
            lat[-1] = -180.0 - lat[-1]
        if (lat1 > 90.0):
            lat[0] = 180. - lat[0]
        if (lat1 < -90.0):
            lat[0] = -180.0 - lat[0]

        for j in range(1,nj):
            lat[j] = lat[0] + float(j)* (lat[-1] - lat[0])/float(nj-1)
            if (lat[j] > 90.0):
                lat[j] = 180.0  - lat[j]
            if (lat[j] < -90.0):
                lat[j] = -180.0 - lat[j]
        if (lat[1] - lat[0] < 0):
            lat = lat[::-1]
        return (lat,lon)

    def ddx(self,s,lat,lon,missingData):

        lonLen = len(lon)
        latLen = len(lat)
        dsdx = np.empty((latLen,lonLen))
        rearth = 6371221.3

        di = abs(np.cos(np.radians(0.0))*rearth*(np.radians(lon[1]-lon[0])))

        # GRIB order - S-N(OUTER)
        #              W-E(INNER)
        for j in range(0,latLen):                
            for i in range(1, lonLen-1):
                if (abs(lat[j]) >= 90.0):
                    dsdx[j,0] = 0.0
                elif (s[j, i+1] > -999.99 and s[j,i-1] > -999.99):
                    dsdx[j, i] = (s[j,i+1] - s[j,i-1])/(2.*di)
                elif (s[j,i+1] < -999.99 and s[j,i-1] > -999.99 and s[j,i] > -999.99):
                    dsdx[j,i] = (s[j,i] - s[j,i-1])/di
                elif (s[j,i+1] > -999.99 and s[j,i-1] < -999.99 and s[j,i] > -999.99):
                    dsdx[j,i]  = (s[j,i+1] - s[j,i])/di
                else:
                    dsdx[j,i] = -999.99
        for j in range(0,latLen):
            if (abs(lat[j]) >= 90.0):
                dsdx[j,0] = 0.0
                dsdx[j,-1] =0.0
            elif (np.allclose(2*lon[0]-lon[-1],lon[1],1e-3) or np.allclose(2*lon[0]-lon[-1],lon[1] + 360.0,1e-3)):
                if (s[j, 1] > -999.99 and s[j, -1] > -999.99):
                    dsdx[j, 0] = (s[j, 1] - s[j,-1]) / (2.*di)
                elif (s[j,1] < -999.99 and s[j,-1] > -999.99 and s[j,0] > -999.99) :
                    dsdx[j,0] = (s[j,1] - s[j,-1]) / di
                elif (s[j, 1] > -999.99 and s[j,-1] > -999.99 and s[j,0] > -999.99):
                    dsdx[j,0] = (s [j,1] - s[j,0]) /di
                else:
                    dsdx[j, 0] = -999.99
                if (s[j,0] > -999.99 and s[j,-2] > -999.99):
                    dsdx[j,-1] = (s[j, 0] - s[j,-2]) / (2. * di)
                elif (s[j,0] < -999.99 and s[j,-2] > -999.99 and s[j,-1] > -999.99):
                    dsdx[j,-1] = (s[j,-1] - s[j,-2]) / di
                elif (s[j,0] > -999.99 and s[j,-2] < -999.99 and s[j,-1] > -999.99) :
                    dsdx[j,-1] = (s[j,1] - s[j,-1]) / di
                else:
                    dsdx[j, -1] = -999.99
            elif (np.allclose(lon[0],lon[-1],1e-3)):
                if (s[j, 1] > -999.99 and s[j,-2] > -999.99) :
                    dsdx[j,0] = (s[j,1] - s[j,-2]) / (2. * di)
                elif (s[j,1] < -999.99 and s[j,-2] > -999.99 and s[j,0] > -999.99) :
                    dsdx[j,0] = (s[j,0] - s[j,-2]) / di
                elif (s[1, j] > -999.99 and s[- 2, j] < -999.99 and s[0, j] > -999.99):
                    dsdx[j,0] = (s[j,1] - s[j,0]) / di
                else:
                    dsdx[j,0] = -999.99
                dsdx[j,-1] = dsdx[j,0]
            else:
                if (s[j, 1] > -999.99 and s[j,0] > -999.99) :
                    dsdx[j,0] = (s[j,1] - s[j,0]) /di
                else:
                    dsdx[j,0] = -999.99
                
                if (s[j,-1] > -999.99 and s[j,-2] > -999.99):
                    dsdx[j,-1] = (s[j,-1] -s[j,-2]) /di
                else:
                    dsdx[j,-1] = -999.99

        return dsdx

    def ddy(self,s,lat,lon):
        lonLen = len(lon)
        latLen = len(lat)
        dsdy = np.empty((latLen,lonLen))
        rearth = 6371221.3
        dj = abs(np.radians((lat[0]-lat[1])) * rearth)
        # North Pole
        
        for i in range(0,lonLen):
            s0 = s[0,i]
            s1 = s[1,i]
            if (s0 -999.99 and s1 > -999.99):
                # Make sure derivative is  dq/dy = [q(north) - q(south)]/dlat
                dsdy[0,i] = (s[0,i] - s[1,i])/dj
            else:
                dsdy[0,i] = -999.99
        dsdyA = s[0,:] > -999.99
        dsdyB = s[1,:] > -999.99
        dsdy1 = np.where(dsdyA & dsdyB,(s[0,:]-s[1,:])/dj,-999.99)

        # South Pole

        dsdyC = s[-1,:] > -999.99
        dsdyD = s[-2,:] > -999.99
        dsdy1 = np.where(dsdyC & dsdyD,(s[-2,:]-s[-1,:])/dj,dsdy1)
        for i in range(0,lonLen):
            if (s[-1,i] > -999.99 and s[-2,i] > -999.99):
                # Make sure derivative is  dq/dy = [q(north) - q(south)]/dlat
                dsdy[-1,i] = (s[-2,i] - s[-1,i])/dj
            else:
                dsdy[-1,i] = -999.99
        #has_left = s[:-2,:] > -999.99
        #has_right = s[:-2,:] > -999.99
        #dsdy1 = np.where(has_left & has_right,(s[:-2,:] - s[:2,:])/(2.*dj),dsdy1)
        #Interior grid point
        for j in range(1,latLen-1):
            for i in range(0,lonLen):
                if (s[j-1,i] > -999.99 and s[j+1,i] > -999.99):
                    # Make sure derivative is  dq/dy = [q(north) - q(south)]/dlat
                    dsdy[j,i] = (s[j+1,i] - s[j-1,i])/(2.0*dj)
                    #print(dsdy[j,i])
                elif (s[j-1,i] < -999.99 and s[j+1,i] > -999.99 and s[j,i] > -999.99):
                    dsdy[j,i] = (s[j+1,i] - s[j,i])/dj
                elif (s[j-1,i] > -999.99 and s[j+1,i] < -999.99 and s[j,i] > -999.99):
                    dsdy[j,i] = (s[j,i] - s[j-1,i])/dj
                else:
                    dsdy[j,i] = -999.99

        return dsdy

    def relvor(self,lat,lon,u,v,dvdx,dudy):
        lonLen = len(lon)
        latLen = len(lat)
        rearth = 6371221.3
        relv = np.empty((latLen,lonLen))
        missing = 0
        # Begin South Pole
        for i in range(0,lonLen):
            if (u[1,i] > -999.99):
                relv[0,0] = u[1,i]
                break
            else:
                missing = i
        for i in range(missing+2,lonLen):
            if (u[1,i] > -999.99):
                relv[0,0] = relv[0,0]+u[1,i]
            else:
                missing = missing +1
        if (abs(lon[0] - lon[-1]) > .0001) :
            if (u[-1,1] > -999.99):
                relv[0,0]= relv[0,0]+u[-1,1]
            else:
                missing = missing +1
            relv[0,0] = relv[0,0]* math.cos(math.radians(lat[1]))/(1.-math.sin (math.radians(lat[1])))/rearth /float(lonLen - missing)
        else:
            relv[0,0] = relv[0,0]* math.cos(math.radians(lat[1]))/(1.-math.sin (math.radians(lat[1])))/rearth /float(lonLen -1- missing)
        for i in range(1,lonLen):
            relv[0,i] = relv[0,0]
        #Begin  North Pole
        missing = 0
        for i in range(0,lonLen):
            if (u[-2,i] > -999.99):
                relv[-1,-1] = u[-2,i]
                break
            else:
                missing = i
        for i in range(missing+2,lonLen):
            if (u[-2,i] > -999.99) :
                relv[-1,-1] = relv[-1,-1]+u[-2,i]
            else:
                missing = missing +1
        if (abs(lon[0] - lon[-1] > .001)):
            if(u[-2,i] > -999.99):
                relv[-1,-1]=relv[-1,-1]+u[-2,-1]
            else:
                missing = missing + 1
            relv[-1,-1]=relv[-1,-1]* np.cos(np.radians(lat[-2]))/(1.-np.sin(np.radians(lat[-2])))/rearth /float(lonLen -1- missing)
        else:
            relv[-1,-1]=relv[-1,-1]* np.cos(math.radians(lat[-2]))/(1.-np.sin (np.radians(lat[-2])))/rearth /float(lonLen - 2-missing)        
        for i in range(0,lonLen-1):
            relv[-1,i] = relv[-1,-1]


        ## Global
        for j in range(1,latLen-1):
            for i in range(0,lonLen):
                if (u[j,i] < -999.99 or dvdx[j,i] < -999.99 or dudy[j,i] < -999.99 ):
                    relv[j,i] = -999.99
                else:
                    relv[j,i] = dvdx[j,i]-dudy[j,i]+(u[j,i]*np.tan(np.radians(lat[j])))/rearth

        return relv

    
    def absvor(self,lat,lon,relv):

        
        lonLen = len(lon)
        latLen = len(lat)
        absv = np.empty((latLen,lonLen))
        omega = 7.29212e-05
        for j in range(0,latLen):
            corl = 2.0 * omega * np.sin(np.radians(lat[j]))
            for i in range(0,lonLen):
                if (relv[j,i] > -999.99):
                    absv[j,i] = relv[j,i]+corl
                else:
                    absv[j,i] = -999.99
        return absv

    def pot(self,tmp,pres):
        cp = 1004.0
        md = 28.9644
        R = 8314.41
        Rd = R/md

        pot = tmp * (100000./pres) ** (Rd/cp)

        return pot
    
    def pvonp(self,ni,nj,lat,lon,pres,pres1,pres2,tmp,tmp1,tmp2,u,u1,u2,v,v1,v2,missingData):

        cp = 1004.
        gravity = 9.80665
        md = 28.9644
        R = 8314.41
        Rd = R/md
        kappa = Rd/cp

        #print(v)
        dvdx = self.ddx(v,lat,lon,missingData)
        dudy = self.ddy(u,lat,lon,missingData)
        #print(dudy)
        relv = self.relvor(lat,lon,u,v,dvdx,dudy)

        absv = self.absvor(lat,lon,relv)

        theta = np.empty_like(tmp,dtype=object)

        theta1 = np.empty_like(tmp1,dtype=object)
        theta2 = np.empty_like(tmp2,dtype=object)
        pres = float(pres)
        pres1 = float(pres1)
        pres2 = float(pres2)
        theta[:,:] = self.pot(tmp[:,:],pres1)
        theta1[:,:] = self.pot(tmp1[:,:],pres1)
        theta2[:,:] = self.pot(tmp2[:,:],pres2)

        dpotdx = self.ddx(theta,lat,lon,missingData)

        dpotdy = self.ddy(theta,lat,lon,missingData)

        lnp1p2 = np.log(pres1/pres2)
        dpi = -1. /(pres1 - pres2)
        pv = np.empty((nj,ni))
        for j in range(0,nj):
            for i in range(0,ni):
                stabl = (theta[j,i]/pres) *(np.log(tmp1[j,i]/tmp2[j,i])/lnp1p2 - kappa)

                du = u1[j,i] - u2[j,i]
                dv = v1[j,i] - v2[j,i]
                dth = theta1[j,i] - theta2[j,i]
                vorcor = (du * dpotdy[j,i] - dv * dpotdx[j,i])/dth
                pv[j,i] = gravity * (absv[j,i] +vorcor) * dth *dpi  *1e6

        for j in range(0,nj):
            if (abs(lat[j] - 90.) < 0.001):
                for i in range(1,ni-1):
                    pv[j,0] = pv[j,0]+pv[j,i]
            if (abs(lon[1] - lon[ni-1]) < 0.001):
                pv[j,0] = pv[j,0]/float(ni-1)
            else:
                pv[j,0] = pv[j,0]+pv[j,-1]
                pv[j,0] = pv[j,0]/float(ni-1)
        k = 0
        m = 0
        for j in range(0,nj):
            for i in range(0,ni):
                print(lon[i],lat[j],pv[j,i])
                #if (lat[j] > 0.0  and pv[j,i] <= 0.):
                #    print(lon[i],lat[j],pv[j,i])
                #    k += 1
                #elif (lat[j] < 0.0  and pv[j,i] >= 0.):
                #    print(lon[i],lat[j],pv[j,i])
                #    m += 1
        #print(k,m)
        return pv

    def pvlayr(self,ni,nj,lat,lon,pres,pres1,pres2,tmp1,tmp2,u1,u2,v1,v2):
        
        cp = 1004.
        gravity = 9.80665
        md = 28.9644
        R = 8314.41
        Rd = R/md
        kappa = Rd/cp

        pres1 = float(pres1)
        pres2 = float(pres2)
        lnp1 = np.log(pres1)
        lnp2 = np.log(pres2)
        pav = (pres1*lnp1+pres2*lnp2)/(lnp1+lnp2)

        uav = np.empty((nj,ni))
        vav = np.empty((nj,ni))
        tav = np.empty((nj,ni))
        potav = np.empty((nj,ni))
        uav[:,:] = (lnp2*u2[:,:] + lnp1*u1[:,:])/(lnp1+lnp2)
        vav[:,:] = (lnp2*v2[:,:] + lnp1*v1[:,:])/(lnp1+lnp2)
        tav[:,:] = np.exp((lnp2*np.log(tmp2[:,:]) + lnp1*np.log(tmp1[:,:]))/(lnp1+lnp2))
        potav[:,:] = self.pot(tav,pav)
        missingData = float(-999.99)
        dvdx = self.ddx(vav,lat,lon,missingData)
        dudy = self.ddy(uav,lat,lon,missingData)
        relv = self.relvor(lat,lon,uav,vav,dvdx,dudy)
        absv = self.absvor(lat,lon,relv)
        dpotdx = self.ddx(potav,lat,lon,missingData)
        dpotdy = self.ddy(potav,lat,lon,missingData)
        pv = np.empty((nj,ni))
        stabl = np.empty((nj,ni))
        vorcor = np.empty((nj,ni))

        for j in range(0,nj):
            for i in range(0,ni):
                stabl[j,i] = potav[j,i]/pav * (np.log(tmp1[j,i]/tmp2[j,i])/(lnp1-lnp2)-kappa)
                vorcor[j,i] = ((v1[j,i] - v2[j,i])*dpotdx[j,i]-(u1[j,i] - u2[j,i])*dpotdy[j,i])/(self.pot(tmp1[j,i],pres1)-self.pot(tmp2[j,i],pres2))
                pv[j,i] = -gravity * (absv[j,i]+vorcor[j,i]) * stabl[j,i] * 1e6

        for j in range(0,nj):
            if (abs(lat[j] - 90.) < 0.001):
                for i in range(1,ni-1):
                    pv[j,0] = pv[j,0]+pv[j,i]
            if (abs(lon[1] - lon[ni-1]) < 0.001):
                pv[j,0] = pv[j,0]/float(ni-1)
            else:
                pv[j,0] = pv[j,0]+pv[j,-1]
                pv[j,0] = pv[j,0]/float(ni-1)
                
        k = 0
        m = 0
        #for j in range(0,nj):
        #    for i in range(0,ni):
                #if (lat[j] > 0.0 and pv[i,j] <= 0.):
                #    print(lat[j])
                #    k += 1
                #elif (lat[j] < 0.0 and pv[i,j] >= 0.):
                #    print(lat[j])
                #    m += 1
        #print(k,m)
        return pv

    def  pv_isobaric_bluestein(self,ni,nj,lat,lon,pres,pres1,pres2,tmp,tmp1,tmp2,u,u1,u2,v,v1,v2,missingData):

        gravity = 9.80665
        pres = float(pres)
        pres1 = float(pres1)
        pres2 = float(pres2)
        missingData = float(-999.99)
        dvdx = self.ddx(v,lat,lon,missingData)
        dudy = self.ddy(u,lat,lon,missingData)
        relv = self.relvor(ni,nj,lat,lon,u,v,dvdx,dudy)
        absv = self.absvor(lat,lon,relv)

        dtdx = self.ddx(tmp,lat,lon,missingData)

        dtdy = self.ddy(tmp,lat,lon,missingData)
        theta = np.empty_like(tmp,dtype=object)
        theta1 = np.empty_like(tmp1,dtype=object)
        theta2 = np.empty_like(tmp2,dtype=object)
        theta[:,:] = self.pot(tmp[:,:],pres)
        theta1[:,:] = self.pot(tmp1[:,:],pres1)
        theta2[:,:] = self.pot(tmp2[:,:],pres2)

        dpi = pres1 - pres2
        pv = np.empty((nj,ni))
        for j in range(0,nj):
            for i in range(0,ni):
                dudp = (u1[j,i] - u2[j,i])/dpi
                dvdp = (v1[j,i] - v2[j,i])/dpi
                dth = theta1[j,i] - theta2[j,i]
                static_stability =-(tmp[j,i]/theta[j,i])*dth/dpi
                vorcor = (dvdp * dtdx[j,i]) - (dudp * dtdy[j,i])*static_stability
                stability = dth/dpi
                pv[j,i] = -gravity *(absv[j,i] +vorcor) * dth/dpi  *10**6

        for j in range(0,nj):
            if (abs(lat[j] - 90.) < 0.001):
                for i in range(1,ni-1):
                    pv[j,0] = pv[j,0]+pv[j,i]
            if (abs(lon[1] - lon[ni-1]) < 0.001):
                pv[j,0] = pv[j,0]/float(ni-1)
            else:
                pv[j,0] = pv[j,0]+pv[j,-1]
                pv[j,0] = pv[j,0]/float(ni-1)

        #xx = pv[np.where(pv > 0.0)]
        #yy = lat[np.where(lat <0.0)]
        #lats = np.reshape(lat,(nj,ni))
        #pv = np.where((pv > 0.0) & (lat < 0.0))
        #print(len(pv))
        #sys.exit()
        return pv
    
    def  pv_rossby(self,ni,nj,lat,lon,pres,pres1,pres2,tmp,tmp1,tmp2,u,u1,u2,v,v1,v2,missingData):

        gravity = 9.80665
        pres = float(pres)
        pres1 = float(pres1)
        pres2 = float(pres2)
        missingData = float(-999.99)
        dvdx = self.ddx(v,lat,lon,missingData)
        dudy = self.ddy(u,lat,lon,missingData)
        relv = self.relvor(ni,nj,lat,lon,u,v,dvdx,dudy)
        absv = self.absvor(lat,lon,relv)
        theta = np.empty_like(tmp,dtype=object)
        theta1 = np.empty_like(tmp1,dtype=object)
        theta2 = np.empty_like(tmp2,dtype=object)
        theta[:,:] = self.pot(tmp[:,:],pres1)
        theta1[:,:] = self.pot(tmp1[:,:],pres1)
        theta2[:,:] = self.pot(tmp2[:,:],pres2)

        dpotdx = self.ddx(theta,lat,lon,missingData)

        dpotdy = self.ddy(theta,lat,lon,missingData)
        dpi = pres1 - pres2
        pv = np.empty((nj,ni))

        for j in range(0,nj):
            for i in range(0,ni):
                dudp = (u1[j,i] - u2[j,i])/dpi
                dvdp = (v1[j,i] - v2[j,i])/dpi
                dth = theta1[j,i] - theta2[j,i]
                stability = dth/dpi
                vorcor = (dvdp*dpotdx[j,i])-(dudp*dpotdy[j,i])
                pv[j,i] = -gravity* (absv[j,i]*stability-vorcor)*10**6
                #print(pv[j,i])

        for j in range(0,nj):
            if (abs(lat[j] - 90.) < 0.001):
                for i in range(1,ni-1):
                    pv[j,0] = pv[j,0]+pv[j,i]
            if (abs(lon[1] - lon[ni-1]) < 0.001):
                pv[j,0] = pv[j,0]/float(ni-1)
            else:
                pv[j,0] = pv[j,0]+pv[j,-1]
                pv[j,0] = pv[j,0]/float(ni-1)
        k = 0
        m = 0
        for j in range(0,nj):
            for i in range(0,ni):
                #print(lon[i],lat[j],pv[j,i])
                if (lat[j] > 0.0  and pv[j,i] <= 0.):
                    #print(lon[i],lat[j],pv[j,i])
                    k += 1
                elif (lat[j] < 0.0  and pv[j,i] >= 0.):
                    #print(lon[i],lat[j],pv[j,i])
                    m += 1
        print(k,m)
        #sys.exit()
        return pv

    def svor(self,lats,lons,u,v):
        latLen = len(lats)
        lonLen = len(lons)

        for k in range(0,17):
            u2d = u[:,:,9]
            v2d = v[:,:,9]
            nc_f1 = './uwnd_200_2020_13_9_00Z.nc'  # Your filename
            nc_f2 = './vwnd_200_2020_13_9_00Z.nc'  # Your filename
            ncu = Dataset(nc_f1,'r')
            uwnd = ncu.variables['uwnd'][:]
            lons = ncu.variables['lon'][:]
            lats = ncu.variables['lat'][:]
            ncu.close()
            ncv = Dataset(nc_f2, 'r')
            vwnd = ncv.variables['vwnd'][:]
            ncv.close()
            print((v2d==vwnd).all())
            uwnd, uwnd_info = prep_data(uwnd, 'tpyx')
            vwnd, vwnd_info = prep_data(vwnd, 'tpyx')
            w1 = VectorWind(u2d,v2d)
            print(uwnd)
            w = VectorWind(uwnd,vwnd)
            absv=w.absolutevorticity()
            print(min(absv.flatten()))
            print(max(absv.flatten()))
            absv=np.squeeze(absv,axis=2)
            absv1 = w1.absolutevorticity()
            #absv1 = np.squeeze(absv1,axis=2)
            #print(np.array_equal(absv,absv1))
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
            clevs = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
            fill = ax.contourf(lons, lats, absv * 1e5, clevs, cmap=plt.cm.RdBu_r,
                               transform=ccrs.PlateCarree(), extend='both')
            ax.coastlines()
            ax.gridlines()
            ax.set_xticks([0, 60, 120, 180, 240, 300, 359.99], crs=ccrs.PlateCarree())
            ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
            lon_formatter = LongitudeFormatter(zero_direction_label=True,
                                               number_format='.0f')
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            plt.colorbar(fill, orientation='horizontal')
            plt.title('Absolute Vorticity ($10^{-11}$s$^{-1}$)', fontsize=16)
            plt.show()
            exit()


        
    def sipv(self,lats,lons,kthta,thta,pthta,uthta,vthta,missingData):

        latLen = len(lats)
        lonLen = len(lons)
        ipv = np.empty((kthta,latLen,lonLen))
        p0 = 100000.
        kappa = 2./7.
        gravity = 9.80665
        rearth = 6371221.3
        ntrunc = int(lonLen/2)
        np.seterr(all='warn')
        warnings.filterwarnings('error')

        #x = Spharm(lonLen,latLen,ntrunc,rearth,gridtype='regular')
        for k in range(0,kthta):
            uthta2d = uthta[k,:,:]
            vthta2d = vthta[k,:,:]
            w = VectorWind(uthta2d,vthta2d)

            absv=w.absolutevorticity()

            for j in range(0,latLen):
                for i in range(0,lonLen):
                    if (k == 0):
                        if (pthta[k,j,i] <= 0. or pthta[k+1,j,i] <= 0. or absv[j,i] < missingData):
                            ipv[k,j,i] = missingData
                        else:
                            tdwn = thta[k] * (pthta[k,j,i]/p0)**kappa
                            tup =  thta[k+1] * (pthta[k+1,j,i]/p0)**kappa
                            dlt = np.log(tup) - np.log(tdwn)
                            if (pthta[k+1,j,i] == pthta[k,j,i]):
                                pthta[k+1,j,i] +=10
                            dlp = np.log(pthta[k+1,j,i])-np.log(pthta[k,j,i])
                            try:
                                dltdlp = dlt/dlp
                            except Warning:
                                print(pthta[k+1,j,i],pthta[k,j,i])
                                print(sys.exc_info())
                                print(dlt,dlp)
                                sys.exit()
                            stabl = (thta[k]/pthta[k,j,i]) *(dltdlp-kappa)
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                    elif (k == kthta-1):
                        if (pthta[k,j,i] <= 0. or pthta[k-1,j,i] <= 0. or absv[j,i] < missingData):
                            ipv[k,j,i] = missingData
                        else:
                            tdwn = thta[k-1] * (pthta[k-1,j,i]/p0)**kappa
                            tup =  thta[k] * (pthta[k,j,i]/p0)**kappa
                            dlt = np.log(tup/tdwn)
                            dlp = np.log(pthta[k,j,i]/pthta[k-1,j,i])
                            dltdlp = dlt/dlp
                            stabl = (thta[k]/pthta[k,j,i]) *(dltdlp-kappa)
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                    else:
                        if (pthta[k+1,j,i] > 0. and pthta[k-1,j,i] > 0. and absv[j,i] > missingData):
                            tdwn = thta[k-1] * (pthta[k-1,j,i]/p0)**kappa
                            tup =  thta[k+1] * (pthta[k+1,j,i]/p0)**kappa
                            dlt = np.log(tup/tdwn)
                            if (pthta[k+1,j,i] == pthta[k-1,j,i]):
                                pthta[k+1,j,i] +=10
                            dlp = np.log(pthta[k+1,j,i]/pthta[k-1,j,i])

                            try:
                                dltdlp = dlt/dlp
                            except Warning:
                                print(dlt,dlp)
                            stabl = (thta[k]/pthta[k,j,i]) *(dltdlp-kappa)
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                            #print(absv[j,i])
                        elif (pthta[k+1,j,i] <= 0. and pthta[k-1,j,i] > 0. and pthta[k,j,i] > 0. and absv[j,i] > missingData):
                            tdwn = thta[k-1]*(pthta[k-1,j,i]/p0)**kappa
                            tup =  thta[k]*(pthta[k,j,i]/p0)**kappa
                            dlt = np.log(tup/tdwn)
                            dlp = np.log(pthta[k,j,i]/pthta[k-1,j,i])
                            dltdlp = dlt/dlp
                            stabl = (thta[k]/pthta[k,j,i]) *(dltdlp-kappa)
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                        elif (pthta[k+1,j,i] > 0. and pthta[k-1,j,i] <= 0. and pthta[k,j,i] > 0. and absv[j,i] > missingData):
                            tdwn = thta[k] * (pthta[k,j,i]/p0)**kappa
                            tup =  thta[k+1] * (pthta[k+1,j,i]/p0)**kappa
                            dlt = np.log(tup/tdwn)
                            dlp = np.log(pthta[k+1,j,i]/pthta[k,j,i])
                            dltdlp = dlt/dlp
                            stabl = (thta[k]/pthta[k,j,i]) *(dltdlp-kappa)
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                        else:
                            ipv[k,j,i] = missingData

            ipv[k,:,:] = ndimage.gaussian_filter(ipv[k,:,:]*1e6,sigma=2,order=0)
        return ipv

    
    def ipv(self,lats,lons,kthta,thta,pthta,uthta,vthta,missingData):

        latLen = len(lats)
        lonLen = len(lons)
        ipv = np.empty((kthta,latLen,lonLen))
        p0 = 100000.
        kappa = 2./7.
        gravity = 9.80665
        for k in range(0,kthta):
            dvdx = self.ddx(vthta[k,:,:],lats,lons,missingData)
            dudy = self.ddy(uthta[k,:,:],lats,lons,missingData)
            relv = self.relvor(lats,lons,uthta[k,:,:],vthta[k,:,:],dvdx,dudy)
            absv = self.absvor(lats,lons,relv)
            #print(absv.shape)
            for j in range(0,latLen):
                for i in range(0,lonLen):
                    if (k == 0):
                        if (pthta[k,j,i] <= 0. or pthta[k+1,j,i] <= 0. or absv[j,i] < missingData):
                            ipv[k,j,i] = missingData
                        else:
                            tdwn = thta[k] * (pthta[k,j,i]/p0)**kappa
                            tup =  thta[k+1] * (pthta[k+1,j,i]/p0)**kappa
                            dlt = np.log(tup/tdwn)
                            dlp = np.log(pthta[k+1,j,i]/pthta[k,j,i])
                            dltdlp = dlt/dlp
                            stabl = (thta[k]/pthta[k,j,i]) *(dltdlp-kappa)
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                    elif (k == kthta-1):
                        if (pthta[k,j,i] <= 0. or pthta[k-1,j,i] <= 0. or absv[j,i] < missingData):
                            ipv[k,j,i] = missingData
                        else:
                            tdwn = thta[k-1] * (pthta[k-1,j,i]/p0)**kappa
                            tup =  thta[k] * (pthta[k,j,i]/p0)**kappa
                            dlt = np.log(tup/tdwn)
                            dlp = np.log(pthta[k,j,i]/pthta[k-1,j,i])
                            dltdlp = dlt/dlp
                            stabl = (thta[k]/pthta[k,j,i]) *(dltdlp-kappa)
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                    else:
                        if (pthta[k+1,j,i] > 0. and pthta[k-1,j,i] > 0. and absv[j,i] > missingData):
                            tdwn = thta[k-1] * (pthta[k-1,j,i]/p0)**kappa
                            tup =  thta[k+1] * (pthta[k+1,j,i]/p0)**kappa
                            dlt = np.log(tup/tdwn)
                            dlp = np.log(pthta[k+1,j,i]/pthta[k-1,j,i])
                            dltdlp = dlt/dlp
                            stabl = (thta[k]/pthta[k,j,i]) *(dltdlp-kappa)
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                            #print(absv[j,i])
                        elif (pthta[k+1,j,i] <= 0. and pthta[k-1,j,i] > 0. and pthta[k,j,i] > 0. and absv[j,i] > missingData):
                            tdwn = thta[k-1]*(pthta[k-1,j,i]/p0)**kappa
                            tup =  thta[k]*(pthta[k,j,i]/p0)**kappa
                            dlt = np.log(tup/tdwn)
                            dlp = np.log(pthta[k,j,i]/pthta[k-1,j,i])
                            dltdlp = dlt/dlp
                            stabl = (thta[k]/pthta[k,j,i]) *(dltdlp-kappa)
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                        elif (pthta[k+1,j,i] > 0. and pthta[k-1,j,i] <= 0. and pthta[k,j,i] > 0. and absv[j,i] > missingData):
                            tdwn = thta[k] * (pthta[k,j,i]/p0)**kappa
                            tup =  thta[k+1] * (pthta[k+1,j,i]/p0)**kappa
                            dlt = np.log(tup/tdwn)
                            dlp = np.log(pthta[k+1,j,i]/pthta[k,j,i])
                            dltdlp = dlt/dlp
                            stabl = (thta[k]/pthta[k,j,i]) *(dltdlp-kappa)
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                        else:
                            ipv[k,j,i] = missingData
            ipv[k,:,:] = ndimage.gaussian_filter(ipv[k,:,:]*1e6,sigma=2,order=0)                            
        return ipv

    def ipv2(self,lats,lons,kthta,thta,pthta,uthta,vthta,missingData):

        latLen = len(lats)
        lonLen = len(lons)
        ipv = np.empty((kthta,latLen,lonLen))
        p0 = 100000.
        kappa = 2./7.
        gravity = 9.80665
        for k in range(0,kthta):
            dvdx = self.ddx(vthta[k,:,:],lats,lons,missingData)
            dudy = self.ddy(uthta[k,:,:],lats,lons,missingData)
            relv = self.relvor(lats,lons,uthta[k,:,:],vthta[k,:,:],dvdx,dudy)
            absv = self.absvor(lats,lons,relv)
            for j in range(0,latLen):
                for i in range(0,lonLen):
                    if (k == 0):
                        if (pthta[k,j,i] <= 0. or pthta[k+1,j,i] <= 0. or absv[j,i] < missingData):
                            ipv[k,j,i] = missingData
                        else:
                            tdwn = thta[k] 
                            tup =  thta[k+1] 
                            deltaTheta = tup - tdwn
                            deltaP = pthta[k+1,j,i]-pthta[k,j,i]
                            stabl = deltaTheta/deltaP
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                    elif (k == kthta-1):
                        if (pthta[k,j,i] <= 0. or pthta[k-1,j,i] <= 0. or absv[j,i] < missingData):
                            ipv[k,j,i] = missingData
                        else:
                            tdwn = thta[k-1] 
                            tup =  thta[k] 
                            deltaTheta = tup - tdwn
                            deltaP = pthta[k,j,i] - pthta[k-1,j,i]
                            stabl = deltaTheta/deltaP
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                    else:
                        if (pthta[k+1,j,i] > 0. and pthta[k-1,j,i] > 0. and absv[j,i] > missingData):
                            tdwn = thta[k-1] 
                            tup =  thta[k+1] 
                            deltaTheta = tup - tdwn
                            deltaP = pthta[k+1,j,i] - pthta[k-1,j,i]
                            stabl = deltaTheta/deltaP
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                            #print(absv[j,i])
                        elif (pthta[k+1,j,i] <= 0. and pthta[k-1,j,i] > 0. and pthta[k,j,i] > 0. and absv[j,i] > missingData):
                            tdwn = thta[k-1]
                            tup =  thta[k]
                            deltaTheta = tup - tdwn
                            deltaP = pthta[k,j,i] - pthta[k-1,j,i]
                            stabl = deltaTheta/deltaP
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                        elif (pthta[k+1,j,i] > 0. and pthta[k-1,j,i] <= 0. and pthta[k,j,i] > 0. and absv[j,i] > missingData):
                            tdwn = thta[k] 
                            tup =  thta[k+1] 
                            deltaTheta = tup - tdwn
                            deltaP = pthta[k+1,j,i] - pthta[k,j,i]
                            stabl = deltaTheta/deltaP
                            ipv[k,j,i] = -gravity * absv[j,i] * stabl
                        else:
                            ipv[k,j,i] = missingData
        return ipv


    
    def p2thta(self,lats,lons,plevs,tsfc,psfc,tpres):

        maxlvl = 17
        plvls = len(plevs)
        cp = float(1004)

        md = 28.9644
        R = 8314.41
        Rd = R/md
        kappa = 2./7.
        dthta = 10.
        epsln = 0.001
        kmax = 10
        p0 = 100000.
        latLen = len(lats)
        lonLen = len(lons)
        
        pthta = np.zeros((plvls,latLen,lonLen))

        thtap = np.zeros((plvls,latLen,lonLen))
        potsfc = np.zeros((latLen,lonLen))
        alogp = np.zeros((plvls))
        thta = np.zeros(maxlvl)


        # Calculate potential temperature at the surface. Keep track of lowest value.

        thtalo = self.pot(tsfc[0,0],psfc[0,0])

        for j in range(0,latLen):
            for i in range(0,lonLen):
                potsfc[j,i] = self.pot(tsfc[j,i],psfc[j,i])
                if (potsfc[j,i] < thtalo):
                    thtalo = potsfc[j,i]
        # Compute potential temperature for each isobaric level eliminating superadiabatic or neutral layer

        thtahi = self.pot(tpres[9,0,0],plevs[9])

        for k in range(0,len(plevs)):
            for j in range(0,latLen):
                for i in range(0,lonLen):
                    thtap[k,j,i] = self.pot(tpres[k,j,i],plevs[k])
                    if (psfc[j,i] > plevs[k]) :
                        if (k > 0):
                            if (psfc[j,i] < plevs[k-1]):
                                if (thtap[k,j,i] <= potsfc[j,i]):
                                    thtap[k,j,i] = potsfc[j,i]+0.01
                            elif (thtap[k,j,i] <= thtap[k-1,j,i]):
                                thtap[k,j,i] = thtap[k-1,j,i]+0.01
                        else:
                            if (thtap[0,j,i] <= potsfc[j,i]):
                                thtap[0,j,i] = potsfc[j,i]+0.01
                        
                        if (k >= 9 and thtap[k,j,i] > thtahi):
                            thtahi = thtap[k,j,i]

        # Identify isentropic levels to interpolate to
        kout = 0
        while (True):
            kout += 1
            thta[0] = 200. + float(kout-1)*dthta
            if (thta[0] + dthta >= thtalo):
                break

        looping = True
        while(looping):
            thta[0] += dthta            
            npts = 0
            j = 0
            while (looping and j < latLen):
                i = 0
                while (looping and i < lonLen):
                    if (potsfc[j,i] <= thta[0]):
                        npts +=1
                        if (npts >= (latLen*lonLen)/10.):
                            looping = False
                    i += 1
                j +=1

        print('first entropic level is ',thta[0])
        kthta = 1
        while (kthta < maxlvl):
            if (thta[kthta] <= thtahi):
                thta[kthta] = thta[kthta-1]+dthta
                kthta += 1
        looping = True

        while (looping):
            kthta -=1
            npts = 0
            j = 0
            while (looping and j < latLen):
                i = 0
                while (looping and i < lonLen):
                    if (thtap[-1,j,i] >= thta[kthta]):
                        npts += 1
                        if (float(npts) >= float(latLen*lonLen)/10.):
                            looping = False
                    i +=1
                j = +1

        if (kthta >= maxlvl):
            print('P2THTA: ONLY THE FIRST')
        else:
            print('TOP ISENTROPIC LEVEL', thta[kthta])

        print(thta)

        alogp[:] = np.log(plevs[:])
        maxit = 0
        resmax = 1.

        for kout in range(0,kthta):
            for j in range(0,latLen):
                for i in range(0,lonLen):
                    # Begin IF
                    if (thta[kout] < potsfc[j,i]):
                        pthta[kout,j,i] = psfc[j,i]
                    elif (thta[kout] > thtap[-1,j,i]):
                        pthta[kout,j,i] = 1.
                    elif (abs(thta[kout]-potsfc[j,i]) < 0.001):
                        pthta[kout,j,i] = psfc[j,i]
                    else:
                        looping = True
                        kin = 0
                        while (looping and kin < plvls):
                            if (thta[kout] < thtap[kin,j,i]):
                                #Begin IF
                                if (kin == 0):
                                    potdwn = potsfc[j,i]
                                    pdwn = psfc[j,i]
                                    alogpd = np.log(psfc[j,i])
                            #       # Begin If
                                    if (abs(psfc[j,i]-plevs[kin]) < 0.001):
                                        potup = thtap[kin+1,j,i]
                                        pup = plevs[kin+1]
                                        alogpu = alogp[kin+1]
                                    else:
                                        potup = thtap[kin,j,i]
                                        pup = plevs[kin]
                                        alogpu = alogp[kin]
                                   # End if
                                elif (potsfc[j,i] > thtap[kin+1,j,i]):
                                    potdwn = potsfc[j,i]
                                    pdwn = psfc[j,i]
                                    alogpd = np.log(psfc[j,i])
                                    if (abs(psfc[j,i]-plevs[kin]) < 0.001):
                                        potup =thtap[kin+1,j,i]
                                        pup = plevs[kin+1]
                                        alogpu = alogp[kin+1]
                                    else:
                                        potup = thtap[kin,j,i]
                                        pup = plevs[kin]
                                        alogpu = alogp[kin]
                                    #End if
                                else:
                                    potup = thtap[kin,j,i]
                                    pup = plevs[kin]
                                    alogpu = alogp[kin]
                                    potdwn = thtap[kin-1,j,i]
                                    pdwn = plevs[kin-1]
                                    alogpd = alogp[kin-1]
                                # End if
                                looping = False
                            kin +=1
                        #  While loop end
                        # End if Matches line 625
                        tdwn = potdwn * (pdwn/100000.)** kappa
                        tup = potup *(pup/100000.)** kappa
                        a = (tup - tdwn)/(alogpu - alogpd)
                        b = tup - a *alogpu
                        if (alogpu-alogpd == 0.):
                            diffpupd = 0.0001
                            try:
                                dltdlp = (np.log(tup/tdwn))/(diffpupd)
                            except Warning:
                                print(dltdlp)
                                print("divide by zero")
                        else:
                            try:
                                dltdlp = (np.log(tup/tdwn))/(alogpu-alogpd)
                            except Warning:
                                print("divide by zero")
                        interc = np.log(tup) - dltdlp*alogpu
                        pln =  (np.log(thta[kout]) - interc - kappa*alogp[0])/(dltdlp-kappa)
                        #pln = alogpd + 0.5 * (alogpu - alogpd)
                        resid = 1
                        #k = 0
                        #pok = (p0)**kappa
                        kmax = 10
                        while (resid  > epsln and k < kmax) :
                            #ekp = np.exp(-kappa * pln)
                            #t = a * pln + b
                            #f = thta[kout] - pok * t * ekp
                            #fp = pok * ekp * (kappa * t  -a)
                            #pin = pln - f/fp
                            #res = abs(pln -pin)
                            #pln = pin
                            #k = k+1
                            t1 = dltdlp * pln+interc
                            thta1 = t1 + kappa *(np.log(p0 / pln))
                            f= np.log(thta[kout]) - thta1
                            dfdp =  dltdlp/np.exp(pln) + kappa/np.exp(-pln)
                            pin = pln - f/dfdp
                            resid = abs(pln - pin)
                            pln = pin
                            k += 1
                        pthta[kout,j,i] = np.exp(pln)
                        #print(pthta[kout,j,i])
                if (pthta[kout-1,j,i] > 0.):
                    if (pthta[kout,j,i] > pthta[kout-1,j,i]):
                        pthta[kout,j,i] = pthta[kout-1,j,i] + 0.01

        ret = []
        ret.append(kthta)
        ret.append(pthta)
        ret.append(thta)
        return ret

    def s2thta(self,lats,lons,pres,kthta,ssfc,psfc,spres,thta,pthta):

        plvls = len(pres)
        latLen = len(lats)
        lonLen = len(lons)
        sthta = np.zeros((kthta,latLen,lonLen))
        lnpu1p = np.zeros((plvls))
        lnpu2p = np.zeros((plvls-1))
        #spres = np.transpose(spres,(2,0,1))

        np.seterr(all='warn')
        warnings.filterwarnings('error')
        
        for kin in range(0,plvls-1):
            lnpu1p[kin] = Decimal(np.log(pres[kin]/pres[kin-1]))
            lnpu2p[kin] = Decimal(np.log(pres[kin]/pres[kin-2]))
            
        lnpu1p[-1] = float(np.log(pres[-1]/pres[-2]))

        for kout in range(0,kthta):
            for j in range(0,latLen):
                for i in range(0,lonLen):
                    if(pthta[kout,j,i] <= 0.):
                        sthta[kout,j,i] = ssfc[j,i]
                    elif (np.allclose(abs(pthta[kout,j,i]-psfc[j,i]),0.001)):
                        sthta[kout,j,i] = ssfc[j,i]
                    else:
                        kin = 0
                        looping = True
                        while (looping and kin < plvls):
                            if (abs(pthta[kout,j,i]-pres[kin]) < 0.001):
                                sthta[kout,j,i] = spres[kin,j,i]
                            elif (pthta[kout,j,i] > pres[kin]):
                                if (kin == 0):
                                    pdwn = psfc[j,i]
                                    sdwn = ssfc[j,i]
                                    if (abs(psfc[j,i]-pres[kin]) < 0.001):
                                        pmid = pres[kin]
                                        pup = pres[kin+1]
                                        smid = spres[kin,j,i]
                                        sup = spres[kin+1,j,i]
                                        lnp1p2 = lnpu1p[kin]
                                        lnp1p3 = float(np.log(pup/pdwn))
                                        lnp2p3 = float(np.log(pmid/pdwn))
                                        #print(pmid,pup)
                                    else:
                                        pmid = pres[kin+1]
                                        pup = pres[kin+2]
                                        smid = spres[kin+1,j,i]
                                        sup = spres[kin+2,j,i]
                                        lnp1p2 = lnpu1p[kin+1]
                                        lnp1p3 = lnpu2p[kin]
                                        lnp2p3 = lnpu1p[kin]
                                        #print(pmid,pup,smid,sup)
                                elif (kin == plvls-1):
                                    pdwn = pres[kin-2]
                                    pmid = pres[kin-1]
                                    pup = pres[kin]
                                    sdwn = spres[kin-2,j,i]
                                    smid = spres[kin-1,j,i]
                                    sup = spres[kin,j,i]
                                    lnp1p2 = lnpu1p[kin-1]
                                    lnp1p3 = lnpu2p[kin-2]
                                    lnp2p3 = lnpu1p[kin-2]
                                elif (psfc[j,i] < pres[kin-1]):
                                    pdwn = psfc[j,i]
                                    sdwn = ssfc[j,i]
                                    if (abs(psfc[j,i] -pres[kin]) > 0.001):
                                        pmid = pres[kin]
                                        pup =  pres[kin+1]
                                        smid = spres[kin,j,i]
                                        sup = spres[kin+1,j,i]
                                        lnp1p2 = lnpu1p[kin]
                                        #lnp1p3 = np.log(pup/pdwn)
                                        if (pup == pdwn):
                                            pup += 10
                                        lnp1p3 = np.log(pup) -np.log(pdwn)
                                        lnp2p3 = float(np.log(pmid/pdwn))
                                    else:
                                        pmid = pres[kin+1]
                                        pup = pres[kin+2]
                                        smid = spres[kin+1,j,i]
                                        sup = spres[kin+2,j,i]
                                        lnp1p2 = lnpu1p[kin+1]
                                        lnp1p3 = lnpu2p[kin]
                                        lnp2p3 = lnpu1p[kin]
                                else:
                                    pdwn = pres[kin-1]
                                    pmid = pres[kin]
                                    pup =  pres[kin+1]
                                    sdwn = spres[kin-1,j,i]
                                    smid = spres[kin,j,i]
                                    sup = spres[kin+1,j,i]
                                    lnp1p2 = lnpu1p[kin]
                                    lnp1p3 = lnpu2p[kin-1]
                                    lnp2p3 = lnpu1p[kin-1]
                                looping = False
                            kin +=1
                        try:
                            qdwn = float(np.log(pthta[kout,j,i]/pmid)*np.log(pthta[kout,j,i]/pup)/lnp2p3/lnp1p3)
                            qmid = float(-np.log(pthta[kout,j,i]/pdwn)*np.log(pthta[kout,j,i]/pup)/lnp2p3/lnp1p2)
                            qup = float(np.log(pthta[kout,j,i]/pdwn)*np.log(pthta[kout,j,i]/pmid)/lnp1p3/lnp1p2)
                            sthta[kout,j,i] = float(qdwn*sdwn + qmid*smid + qup *sup)
                        except Warning:
                            print(qmid,qup,sthta[kout,j,i])
                            print(sys.exc_info())
                            sys.exit()
        return sthta
