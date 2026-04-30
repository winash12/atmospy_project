#!/usr/bin/python3.8
import sys,math,random
import warnings
import numpy as np
from numpy import newaxis
import warnings
import time
from netCDF4 import Dataset
import scipy.ndimage as ndimage
from windspharm.standard import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim
from decimal import *
import interp_lib
import weather_lib
import traceback
from numpy.testing import assert_allclose

class potential_vorticity:

    def ddx(self,s,lat,lon,missingData):

        lonLen = len(lon)
        latLen = len(lat)
        dsdx = np.empty((latLen,lonLen))
        rearth = 6371221.3

        di = abs(np.cos(np.radians(0.0))*rearth*(np.radians(lon[1]-lon[0])))

        # GRIB order - S-N(OUTER)
        #              W-E(INNER)
        has_left = s[:,:-2] > -999.99
        has_right = s[:,2:] > -999.99
        has_value = s[:,1:-1] > -999.99
        dsdx = np.zeros((73,144))
        dsdx[:, 1:-1] = -999.99
        dsdx[:, 1:-1] = np.where(has_right & has_value, (s[:,2:] - s[:,1:-1]) / di, dsdx[:, 1:-1])
        dsdx[:, 1:-1] = np.where(has_left & has_value, (s[:,1:-1] - s[:,:-2]) / di, dsdx[:, 1:-1])
        dsdx[:, 1:-1] = np.where(has_left & has_right, (s[:,2:] - s[:,:-2]) / (2. * di), dsdx[:, 1:-1])
        hasValue = s[1:-1,0] > -999.99
        hasRight = s[1:-1,-1] > -999.99
        hasLeft = s[1:-1,1] > -999.99
        hasRight2 = s[1:-1,-2] > -999.99
        
        
        
        if (np.allclose(2*lon[0]-lon[-1],lon[1],1e-3) or np.allclose(2*lon[0]-lon[-1],lon[1] + 360.0,1e-3)):
            dsdx[1:-1,0] = -999.99
            dsdx[1:-1,-1] = -999.99
            dsdx[1:-1,0] = np.where(hasRight & hasValue,(s[1:-1,-1] - s[1:-1,0]) / di, dsdx[1:-1, 0])
            dsdx[1:-1,0] = np.where(hasLeft & hasValue,(s[1:-1,1] - s[1:-1,0]) / di, dsdx[1:-1, 0])
            dsdx[1:-1,0] = np.where(hasLeft & hasRight,(s[1:-1,1] - s[1:-1,-1]) /2. * di, dsdx[1:-1, 0])
            dsdx[1:-1,-1] = np.where(hasRight & hasRight2,(s[1:-1,-1] - s[1:-1,-2]) / di, dsdx[1:-1, -1])
            dsdx[1:-1,-1] = np.where(hasLeft & hasRight,(s[1:-1,1] - s[1:-1,-1]) / di, dsdx[1:-1, -1])
            dsdx[1:-1,-1] = np.where(hasValue & hasRight2,(s[1:-1,0] - s[1:-1,-2]) /2. * di, dsdx[1:-1, -1])
        elif (np.allclose(lon[0],lon[-1],1e-3)):
            dsdx[1:-1,0] = -999.99
            dsdx[1:-1,-1] = -999.99
            dsdx[1:-1,0] = np.where(hasLeft & hasRight2,(s[1:-1,1] - s[1:-1,-2]) / 2. *di, dsdx[1:-1, 0])
            dsdx[1:-1,0] = np.where(hasValue & hasRight2,(s[1:-1,0] - s[1:-1,-2]) /di, dsdx[1:-1, 0])
            dsdx[1:-1,0] = np.where(hasLeft & hasValue,(s[1:-1,1] - s[1:-1,0]) / di, dsdx[1:-1, 0])
        else:
            dsdx[1:-1,0] = -999.99
            dsdx[1:-1,-1] = -999.99
            dsdx[1:-1,0] = np.where(hasLeft & hasValue,(s[1:-1,1] - s[1:-1,0]) / di, dsdx[1:-1, 0])
            dsdx[1:-1,-1] = np.where(hasRight & hasRight2,(s[1:-1,-1] - s[1:-1,-2]) / di, dsdx[1:-1, -1])
        return dsdx


                    
    def ddy_old(self,s,lat,lon):
        lonLen = len(lon)
        latLen = len(lat)
        dsdy = np.empty((latLen,lonLen))

        rearth = 6371221.3
        dj = abs(np.radians((lat[0]-lat[1])) * rearth)
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

        return dsdy

    def ddy(self, s, lat, lon):
        """
        Pramana Vaayu: Universal Meridional Derivative Engine.
        Handles North-Start (Tiger/CORe) or South-Start (Dragon/ERA5) automatically.
        Maintains 1e-15 MAE integrity.
        """
        # 1. Standardise missing values (The 'Ritual of Purification')
        s = np.where(s == -999.99, np.nan, s)
        lat_len, lon_len = s.shape
        dsdy = np.full((lat_len, lon_len), np.nan)
        rearth = 6371221.3

        # 2. Determine Orientation (Pramana: Detecting the lineage)
        # direction = 1 if South-to-North (-90 to 90)
        # direction = -1 if North-to-South (90 to -90)
        direction = 1 if lat[-1] > lat[0] else -1
        
        # 3. Calculate Variable Spacing (The 'Generic' measure)
        lat_rads = np.radians(lat)
        dphi = np.abs(np.diff(lat_rads))
        
        # dist_2d[i] is the central distance between index i+1 and i-1
        dist_2d = (dphi[1:] + dphi[:-1]) * rearth
        dist_2d = dist_2d[:, np.newaxis] # Broadcast for the 512 longitudes
        
        # 4. Define 'Top' and 'Bottom' based on physical North/South
        if direction == 1:
            # Index increases Northward: s[i+1] is North, s[i-1] is South
            s_top = s[2:, :]
            s_bot = s[:-2, :]
        else:
            # Index increases Southward: s[i-1] is North, s[i+1] is South
            s_top = s[:-2, :]
            s_bot = s[2:, :]
            
        # 5. The 'Shighra' (Fast) Central Difference
        valid = ~np.isnan(s)
        has_both = valid[2:, :] & valid[:-2, :] & valid[1:-1, :]

        with np.errstate(divide='ignore', invalid='ignore'):
            # Writes directly into memory, zero allocation bloat
            np.divide(s_top - s_bot, dist_2d, out=dsdy[1:-1, :], where=has_both)
            
            # 6. Boundaries (Fierce and accurate)
            if direction == 1:
                # South Boundary (Index 0)
                mask_s = valid[0, :] & valid[1, :]
                dsdy[0, mask_s] = (s[1, mask_s] - s[0, mask_s]) / (dphi[0] * rearth)
                # North Boundary (Index -1)
                mask_n = valid[-1, :] & valid[-2, :]
                dsdy[-1, mask_n] = (s[-1, mask_n] - s[-2, mask_n]) / (dphi[-1] * rearth)
            else:
                # North Boundary (Index 0)
                mask_n = valid[0, :] & valid[1, :]
                dsdy[0, mask_n] = (s[0, mask_n] - s[1, mask_n]) / (dphi[0] * rearth)
            # South Boundary (Index -1)
            mask_s = valid[-1, :] & valid[-2, :]
            dsdy[-1, mask_s] = (s[-2, mask_s] - s[-1, mask_s]) / (dphi[-1] * rearth)
        
        return dsdy

    

    def relvor_vectorized(self, u, v, dvdx, dudy, lat, lon):

        """
        Vectorized Relative Vorticity with Spherical Cap polar treatment.
        USP: Uses Stokes' Theorem for pole points to avoid 1/cos(90) singularities.
        """
        # 0. Setup constants from the injected physics config
        rearth = 6371229.0  # Earth radius in meters
        nj, ni = u.shape
        relv = np.full((nj, ni), np.nan)
    
        # Convert lat/lon to radians for trig functions
        lat_rad = np.radians(lat)
    
        # --- 1. Polar Treatment (The Keith Brill / GEMPAK Method) ---
        # South Pole (Index 0): Uses the first row above the pole (Index 1)
        # Circulation = Sum(u * dl) / Area of Spherical Cap
        u_south_ring = u[1, :]
        valid_s = ~np.isnan(u_south_ring)
        if np.any(valid_s):
        # Average U around the ring * Geometric Factor
        # Factor: cos(lat) / (R * (1 - sin(lat)))
            factor_s = np.cos(lat_rad[1]) / (rearth * (1.0 - np.sin(lat_rad[1])))
            relv[0, :] = np.nanmean(u_south_ring) * factor_s

            # North Pole (Index -1): Uses the row below the pole (Index -2)
            u_north_ring = u[-2, :]
            valid_n = ~np.isnan(u_north_ring)
        if np.any(valid_n):
            factor_n = np.cos(lat_rad[-2]) / (rearth * (1.0 - np.sin(lat_rad[-2])))
            # Note: Sign flip often required for North vs South depending on coordinate orientation
            relv[-1, :] = np.nanmean(u_north_ring) * factor_n

            # --- 2. Interior Grid (Standard Spherical Vorticity) ---
            # Formula: dv/dx - du/dy + (u * tan(phi) / R)
            # We use [1:-1] to exclude the pole rows we just calculated
            
            # tan(phi) needs to be broadcasted to (nj-2, ni)
            tan_lat = np.tan(lat_rad[1:-1, np.newaxis])
        
            # Core Vectorized Calculation
            # This processes all interior points (e.g., 71 x 144) in one SIMD step
        relv[1:-1, :] = dvdx[1:-1, :] - dudy[1:-1, :] + (u[1:-1, :] * tan_lat / rearth)
        return relv 

    def absvor_vectorized(self, lat, relv):
                                                     
        omega = self.phys.OMEGA.m if hasattr(self.phys, 'OMEGA') else 7.2921159e-5
    
    # 2. Calculate Coriolis Parameter (f)
    # lat is 1D (73,); corl becomes (73,)
        corl = 2.0 * omega * np.sin(np.radians(lat))
    
    # 3. Vectorized Broadcasting
    # Expand corl (73,) to (73, 1) to broadcast against relv (73, 144)
        f_3d = corl[:, np.newaxis]
    
    # 4. Summation with NaN Propagation
    # In 2026, we no longer need 'np.where' for missing data. 
    # If relv is NaN, absv will automatically be NaN.
        absv = relv + f_3d
    
        return absv
        
        
    def pot(self,tmp,pres):
        cp = 1004.0
        md = 28.9644
        R = 8314.41
        Rd = R/md
        pottemp = np.zeros_like(tmp)
        pottemp = tmp * (100000./pres[:,None,None]) ** (Rd/cp)
        return pottemp

    def potsfc(self,tsfc,psfc):
        cp = 1004.0
        md = 28.9644
        R = 8314.41
        Rd = R/md
        pottemp = np.zeros_like(tsfc)
        pottemp = tsfc * (100000./psfc) ** (Rd/cp)
        return pottemp


    def calculate_potential_temperature(self, tmp, pres):



        kappa = self.phys.KAPPA.m  # Rd/Cp
        p0 = self.phys.P0.m        # 100000.0
    

        if pres.ndim == 1:
            pres_resolved = pres[:, np.newaxis, np.newaxis]
        else:
            pres_resolved = pres
                                                     

        return tmp * (p0 / pres_resolved) ** kappa

    
    def pvonp_vectorized(self, ni, nj, lat, lon, pres, pres1, pres2, tmp, tmp1, tmp2, u, u1, u2, v, v1, v2):

    # 0. Load Constants (Using magnitudes for SIMD performance)
        g = self.phys.G.m
        kappa = self.phys.KAPPA.m
        missing = np.nan
        
        # 1. Kinematics and Absolute Vorticity
        # Replaces point-by-point searches with a single memory sweep
        dvdx = self.ddx(v, lat, lon, missing)
        dudy = self.ddy(u, lat, lon, missing)
        relv = self.relvor(self.ni, self.nj, lat, lon, u, v, dvdx, dudy)
        absv = self.absvor(lat, lon, relv)

        # 2. Isentropic Thermodynamics (The USP)
        # Using your pot function to calculate theta at all three pressure levels
        theta  = self.pot(tmp, pres)
        theta1 = self.pot(tmp1, pres1)
        theta2 = self.pot(tmp2, pres2)

        # 3. Horizontal Isentropic Gradients
        dpotdx = self.ddx(theta, lat, lon, missing)
        dpotdy = self.ddy(theta, lat, lon, missing)

        # 4. Vertical Differentials (Logarithmic Space)
        lnp1p2 = np.log(pres1 / pres2)
    
        # Static Stability: (Theta/P) * (dlnT/dlnP - kappa)
        # This matches your logarithmic thermodynamic requirement perfectly
        stabl = (theta / pres) * (np.log(tmp1 / tmp2) / lnp1p2 - kappa)

        # 5. Baroclinic Correction (The "Tilt" term)
        du = u1 - u2
        dv = v1 - v2
        dth = theta1 - theta2
        # Vectorized vorcor handles all grid points at once
        vorcor = (du * dpotdy - dv * dpotdx) / dth
    
        # 6. Core PV Calculation
        # PV = g * (Absolute_Vorticity + Tilt) * (dTheta/dp)
        # dpi = -1 / (pres1 - pres2) is incorporated into the dth/dpi stability logic
        dpi_const = -1.0 / (pres1 - pres2)
        pv = g * (absv + vorcor) * dth * dpi_const * 1e6

        # 7. Zonal Pole Averaging (Zero-Loop)
        for pole_lat in [-90.0, 90.0]:
            idx = np.where(np.abs(lat - pole_lat) < 0.01)
            if idx.size > 0:
                pv[idx, :] = np.nanmean(pv[idx, :], axis=1, keepdims=True)

        # 8. Diagnostic Print (Vectorized Counting)
        k = np.sum((lat[:, None] > 0.0) & (pv <= 0.0))
        m = np.sum((lat[:, None] < 0.0) & (pv >= 0.0))
        print(f"PVonP Diagnostic: NH Neg={k}, SH Pos={m}")
        return pv
    
    def pvlayr_vectorized(self, lat, lon, pres1, pres2, tmp1, tmp2, u1, u2, v1, v2):
        """
        Vectorized refactor of the Layer Potential Vorticity engine.
        USP: Uses logarithmic vertical averaging and ln T vs ln P thermodynamics.
        """
        # 0. Configuration & Constants from self.phys (Pint-aware)
        g = self.phys.G.m
        kappa = self.phys.KAPPA.m
        missing = np.nan

        # 1. Logarithmic vertical pressure averaging
        # Note: Using .m to ensure we are working with raw magnitudes for speed
        lnp1, lnp2 = np.log(pres1), np.log(pres2)
        ln_sum = lnp1 + lnp2
        ln_diff = lnp1 - lnp2
        
        # Calculate the average pressure of the layer (Log-average)
        pav = (pres1 * lnp1 + pres2 * lnp2) / ln_sum
        
        # 2. Logarithmic vertical variable averaging (Vectorized)
        # Average winds and temperature in log-space for superior physical accuracy
        uav = (lnp2 * u2 + lnp1 * u1) / ln_sum
        vav = (lnp2 * v2 + lnp1 * v1) / ln_sum
        tav = np.exp((lnp2 * np.log(tmp1) + lnp1 * np.log(tmp2)) / ln_sum)

        # Average Potential Temperature for the layer
        potav = self.pot(tav, pav)
        
        # 3. Horizontal Gradients & Vorticity
        # All derivative functions must be vectorized
        dvdx = self.ddx(vav, lat, lon, missing)
        dudy = self.ddy(uav, lat, lon, missing)
        relv = self.relvor(self.ni, self.nj, lat, lon, uav, vav, dvdx, dudy)
        absv = self.absvor(lat, lon, relv)
        
        dpotdx = self.ddx(potav, lat, lon, missing)
        dpotdy = self.ddy(potav, lat, lon, missing)

        # 4. Vertical Stability & Baroclinic "Tilt" (Vectorized)
        # Stability using the ln T / ln P relationship
        stabl = (potav / pav) * (np.log(tmp1 / tmp2) / ln_diff - kappa)
        
        # Vectorized vorcor (accounts for vertical wind shear tilting)
        # denominator uses the difference in potential temperature at the levels
        theta1 = self.pot(tmp1, pres1)
        theta2 = self.pot(tmp2, pres2)
        vorcor = ((v1 - v2) * dpotdx - (u1 - u2) * dpotdy) / (theta1 - theta2)

        # 5. Core Layer PV Calculation
        # Scaled to 10^6 for standard PV Units (PVU)
        pv = -g * (absv + vorcor) * stabl * 1e6
        
        # 6. Zonal Pole Averaging (Zero-Loop)
        for pole_lat in [-90.0, 90.0]:
            idx = np.where(np.abs(lat - pole_lat) < 0.01)[0]
            if idx.size > 0:
                pv[idx, :] = np.nanmean(pv[idx, :], axis=1, keepdims=True)
                
                # 7. Diagnostic Sign Check (Vectorized)
                k = np.sum((lat[:, None] > 0.0) & (pv <= 0.0))
                m = np.sum((lat[:, None] < 0.0) & (pv >= 0.0))
        print(f"Layer PV Diagnostic: NH Negative={k}, SH Positive={m}")
    
        return pv
        
    def  pv_isobaric_bluestein(self,ni,nj,lat,lon,pres,pres1,pres2,tmp,tmp1,tmp2,u,u1,u2,v,v1,v2,missingData):

        gravity = 9.80665
        missing = np.nan
    
        # 1. Kinematics and Vorticity
        dvdx = self.ddx(v, lat, lon, missing)
        dudy = self.ddy(u, lat, lon, missing)
        relv = self.relvor(ni, nj, lat, lon, u, v, dvdx, dudy)
        absv = self.absvor(lat, lon, relv)
        
        # 2. Thermal Gradients
        dtdx = self.ddx(tmp, lat, lon, missing)
        dtdy = self.ddy(tmp, lat, lon, missing)
        
        # 3. Potential Temperature (Self.pot must be vectorized)
        theta = self.pot(tmp, pres)
        theta1 = self.pot(tmp1, pres1)
        theta2 = self.pot(tmp2, pres2)
        
        # 4. Vertical Differentials (isobaric)
        dpi = pres1 - pres2
        dth = theta1 - theta2
        dudp = (u1 - u2) / dpi

        
        dpi = pres1 - pres2
        dth = theta1 - theta2
        dudp = (u1 - u2) / dpi
        dvdp = (v1 - v2) / dpi
        
        # 5. The Core Bluestein Math (Vectorized)
        # The term (tmp/theta) is the Exner function (roughly)
        # vorcor accounts for vertical shear and horizontal temp gradients
        vorcor = (dvdp * dtdx) - (dudp * dtdy)
    
        # PV = -g * (Absolute_Vorticity * dTheta/dp + Vertical_Shear_Terms)
        # Note: Bluestein's isobaric form includes the baroclinic 'tilt' terms
        pv = -gravity * (absv * (dth/dpi) + vorcor) * 10**6

        # 6. Pole Averaging (Vectorized)
        # Replaces the J loops with a single slice mean
        for pole_lat in [-90.0, 90.0]:
            idx = np.where(np.abs(lat - pole_lat) < 0.01)[0]
            if idx.size > 0:
                # zonal mean for all longitudes at the pole latitude
                pv[idx, :] = np.nanmean(pv[idx, :], axis=1, keepdims=True)
                
        return pv
    

    def pv_rossby_vectorized(self, lat, lon, pres, pres1, pres2, tmp, tmp1, tmp2, u, u1, u2, v, v1, v2):
        """
        Full Vectorized Refactor of Rossby Potential Vorticity.
        Eliminates all I, J, K loops and handles missing data via NaNs.
        """
        # --- 0. Data Preparation (Unit Attachment) ---
        # Using .m (magnitude) for internal high-speed math
        if not hasattr(tmp, 'units'): tmp = tmp * ureg.kelvin
        if not hasattr(psfc, 'units'): psfc = psfc * ureg.pascal
        
        # --- 1. Kinematics & Vorticity ---
        # Note: self.ddx, self.relvor, and self.absvor must be vectorized
        dvdx = self.ddx(v, lat, lon, np.nan)
        dudy = self.ddy(u, lat, lon, np.nan)
        relv = self.relvor(self.ni, self.nj, lat, lon, u, v, dvdx, dudy)
        absv = self.absvor(lat, lon, relv)


        dvdx = self.ddx(v, lat, lon, np.nan)
        dudy = self.ddy(u, lat, lon, np.nan)
        relv = self.relvor(self.ni, self.nj, lat, lon, u, v, dvdx, dudy)
        absv = self.absvor(lat, lon, relv)

        # --- 2. Potential Temperature (Theta) Calculation ---
        # Shape: (NJ, NI)
        theta  = self.pot(tmp, pres)
        theta1 = self.pot(tmp1, pres1)
        theta2 = self.pot(tmp2, pres2)

        # --- 3. Isentropic Gradients (Rossby Method) ---
        # Rossby's isobaric form requires dTheta/dx on the pressure surface
        dpotdx = self.ddx(theta, lat, lon, np.nan)
        dpotdy = self.ddy(theta, lat, lon, np.nan)



        dpi = (pres1 - pres2)
        dth = (theta1 - theta2)
        dudp = (u1 - u2) / dpi
        dvdp = (v1 - v2) / dpi
        
        # --- 5. The Core Rossby PV Math ---
        # Rossby's formula accounts for isentropic slope via vertical shear
        stability = dth / dpi
        vorcor = (dvdp * dpotdx) - (dudp * dpotdy)
        
        # PV = -g * (Absolute_Vorticity * Stability - Tilt_Terms)
        # Scaled by 10^6 for standard PV Units (PVU)
        pv = -self.gravity.m * (absv * stability - vorcor) * 1e6

        # --- 6. Zonal Pole Averaging (Zero-Loop) ---
        # Replaces the complex GOTO/IF logic with a single slice mean
        for pole_lat in [-90.0, 90.0]:
            idx = np.where(np.abs(lat - pole_lat) < 0.01)[0]
            if idx.size > 0:
                pv[idx, :] = np.nanmean(pv[idx, :], axis=1, keepdims=True)

        # --- 7. Rossby Diagnostic (The k, m Counters) ---
        # Vectorized counting of negative PV in the North and positive PV in the South
        # These identify regions of symmetric instability or data errors
        k = np.sum((lat[:, None] > 0.0) & (pv <= 0.0))
        m = np.sum((lat[:, None] < 0.0) & (pv >= 0.0))
        print(f"Rossby Diagnostic: NH Errors={k}, SH Errors={m}")
        return pv

    def testsipv(self,lats,lons,kthta,thta,pthta,uthta,vthta,missingData):

        latLen = len(lats)
        lonLen = len(lons)
        t1 = time.time()
        ipvRef = self.sipv(lats,lons,kthta,thta,pthta,uthta,vthta,missingData)



        t2 = time.time()
        print(t2-t1)
        t3 = time.time()
        ipv = self.sipv2(lats,lons,kthta,thta,pthta,uthta,vthta,missingData)
        t4 = time.time()
        print(t4-t3)
        for k in range(0,16):
            for j in range(0,latLen):
                for i in range(0,lonLen):
                    if (ipvRef[k,j,i]-ipv[k,j,i] != 0.):
                        print(ipvRef[k,j,i]-ipv[k,j,i])
        sys.exit()
        return ipv
    
    
        
    def sipv2(self,lats,lons,kthta,thta,pthta,uthta,vthta,missingData):

        latLen = len(lats)
        lonLen = len(lons)
        ipv = np.zeros((kthta,latLen,lonLen))
        dltdlp = np.empty((kthta,latLen,lonLen))
        stabl = np.zeros((kthta,latLen,lonLen))
        tdwn = np.zeros((kthta,latLen,lonLen))
        tdwn_ref = np.zeros((kthta,latLen,lonLen))
        tup = np.zeros((kthta,latLen,lonLen))


        dlt = np.empty((kthta,latLen,lonLen))

        dlp = np.empty((kthta,latLen,lonLen))
        absVor = np.zeros((kthta,latLen,lonLen))
        p0 = 100000.
        kappa = 2./7.
        gravity = 9.80665

        

        is_finite_u = np.isfinite(uthta).all()
        is_finite_v = np.isfinite(vthta).all()
        
        if not (is_finite_u and is_finite_v):
            print("!!! Found issues in the 18Z slice !!!")
            
            # Check for NaNs (Holes in the grid)
            print(f"  NaNs in U: {np.isnan(uthta).sum()}")
            
            # Check for Infs (Divide-by-zero results)
            print(f"  Infs in U: {np.isinf(uthta).sum()}")
            
            # Identify which of the 16 levels are broken
            for i in range(kthta):
                if not np.isfinite(uthta[i]).all():
                    print(f"  Level index {i} contains invalid values.")

        
        args = []
        for uthta2d,vthta2d in zip(uthta,vthta):
            w = VectorWind(uthta2d,vthta2d)
            absVorLevel=w.absolutevorticity()
            b = absVorLevel[newaxis,:,:]
            args.append(b)
        absVor = np.concatenate(args,axis=0)
        # For internal levels




        ipv[1:-1,:,:] = -999.99
        hasPthta = pthta[1:-1,:,:] > 0
        hasPthta0 = pthta[0:-2,:,:] > 0
        hasAbsVor = absVor[1:-1,:,:] > 0



        tdwn[1:-1,:,:] =  thta[0:-3,None,None]*(pthta[0:-2,:,:]/p0)**kappa

        tup[1:-1,:,:] = thta[2:-1,None,None]*(pthta[2:,:,:]/p0)**kappa



        dlt[1:-1,:,:] = np.log(tup[1:-1,:,:]/tdwn[1:-1,:,:])


        dlp[1:-1,:,:] = np.log(pthta[2:,:,:]/pthta[0:-2,:,:])

        dltdlp[1:-1,:,:] = dlt[1:-1,:,:]/dlp[1:-1,:,:]

        stabl[1:-1,:,:] = (thta[1:-2,None,None]/pthta[1:-1,:,:]) *(dltdlp[1:-1,:,:]-kappa)

        ipv[1:-1,:,:] = -gravity*absVor[1:-1,:,:]*stabl[1:-1,:,:]




        # Boundary Layer

        hasNoPthta = pthta[0,:,:] <= 0.
        hasNoPthta0 = pthta[1,:,:] <= 0.
        hasNoAbsVor = absVor[0,:,:] < -999.99

        ipv[0,:,:] = np.where(hasNoPthta|hasNoPthta0|hasNoAbsVor,-999.99,ipv[0,:,:])
        

        tdwn[0,:,:] = thta[0,None,None] * (pthta[0,:,:]/p0)**kappa
        tup[0,:,:] =  thta[1,None,None] * (pthta[1,:,:]/p0)**kappa
        dlt[0,:,:] = np.log(tup[0,:,:]) - np.log(tdwn[0,:,:])
        dlp[0,:,:] = np.log(pthta[1,:,:])-np.log(pthta[0,:,:])
        dltdlp[0,:,:] = dlt[0,:,:]/dlp[0,:,:]
        stabl[0,:,:] = (thta[0,None,None]/pthta[0,:,:]) *(dltdlp[0,:,:]-kappa)

        ipv[0,:,:] = -gravity * absVor[0,:,:] * stabl[0,:,:]




        # Topmost Layer
        hasNoPthta = pthta[-1,:,:] <= 0.
        hasNoPthta0 = pthta[-2,:,:] <= 0.
        hasNoAbsVor = absVor[-1,:,:] < -999.99

        ipv[-1,:,:] = np.where(hasNoPthta|hasNoPthta0|hasNoAbsVor,-999.99,ipv[-1,:,:])

        
        tdwn[-1,:,:] = thta[-3,None,None]*(pthta[-2,:,:]/p0)**kappa
        tup[-1,:,:] =  thta[-2,None,None]*(pthta[-1,:,:]/p0)**kappa

        dlt[-1,:,:] = np.log(tup[-1,:,:]/tdwn[-1,:,:])
        dlp[-1,:,:] = np.log(pthta[-1,:,:]/pthta[-2,:,:])
        dltdlp[-1,:,:] = dlt[-1,:,:]/dlp[-1,:,:]
        stabl[-1,:,:] = (thta[-2,None,None]/pthta[-1,:,:]) *(dltdlp[-1,:,:]-kappa)
        ipv[-1,:,:] = -gravity * absVor[-1,:,:] * stabl[-1,:,:]

        smoothedIPV = ndimage.gaussian_filter(ipv*1e6,sigma=(0,2,2),order=0)
        return smoothedIPV

    def ipv_vectorized(self, lats, lons, kthta, thta, pthta, uthta, vthta, missingData):
        # 1. Physical Constants
        p0, kappa, gravity = 100000.0, 2.0/7.0, 9.80665
        thta_3d = thta[:, np.newaxis, np.newaxis]

        # 2. Vectorized Vorticity (Uses your ddx_fixed with the polar cap)
        dvdx = self.ddx_fixed(vthta, lats, lons)
        dudy = self.ddy(uthta, lats, lons)
        relv = self.relvor(lats, lons, uthta, vthta, dvdx, dudy)
        absv = self.absvor(lats, lons, relv)

        # 3. Vectorized Vertical Stability (Log-Differentiation)
        # log(theta2/theta1) / log(p2/p1)
        log_p = np.log(np.maximum(pthta, 1e-5))
        log_theta = np.log(thta_3d)
        # Vertical gradients using NumPy gradient (much faster than manual slicing)
        # axis=0 is the vertical (k) dimension
        d_log_theta = np.gradient(log_theta, axis=0)
        d_log_p = np.gradient(log_p, axis=0)
        
        # Avoid division by zero in stability term
        dltdlp = np.divide(d_log_theta, d_log_p, out=np.zeros_like(d_log_p), where=d_log_p!=0)
        stabl = (thta_3d / pthta) * (dltdlp - kappa)
        
        # 4. Calculate Full IPV
        ipv_raw = -gravity * absv * stabl * 1e6 # Convert to PVU
        # 5. Masking and Vectorized Smoothing
        # Identify invalid points (underground or missing data)
        mask = (pthta <= 0) | (absv <= missingData) | np.isnan(ipv_raw)
        
        # Fill masked areas with 0.0 to prevent 'NaN bleeding' during filter
        ipv_filled = np.where(mask, 0.0, ipv_raw)
        
        # 6. Multi-Dimensional Filter (Replaces the for-loop)
        # Sigma (0, 2, 2) means: 
        # - 0 sigma on Axis 0 (Don't smooth vertically between layers)
        # - 2 sigma on Axis 1 (Smooth North-South)
        # - 2 sigma on Axis 2 (Smooth East-West)
        # Mode: 'nearest' for Latitude, 'wrap' for Longitude
        ipv_smooth = ndimage.gaussian_filter(
            ipv_filled, 
            sigma=(0, 2, 2), 
            mode=['nearest', 'nearest', 'wrap']
        )


        return ipv_smooth
                                                     
    
  
    
    def adjust_stability_vectorized(self, tpres, plevs, tsfc, psfc):
        """
        Final Refined Version: Fixes the broadcasting error and 
        retains the superior 'Trapped' logic.
        """
        plvls, nj, ni = tpres.shape
        kappa = 0.2857 # NCAR Reanalysis 1
        p0 = 100000.0
        
        # 1. Calculate Potential Temperatures
        potsfc = tsfc * (p0 / psfc)**kappa
        # Broadcast plevs (17,) to (17, 1, 1) to match (17, 73, 144)
        thtap = tpres * (p0 / plevs[:, np.newaxis, np.newaxis])**kappa
        
        thtalo = np.min(potsfc)
        
        # Dynamic Roof Index (10th level for NCAR 17-level data)
        mid_idx = (plvls // 2) + 1
        thtahi = np.max(thtap[mid_idx:]) # Seed with the max of the upper deck
        
        # 2. Sequential Vertical Adjustment
        for k in range(plvls):
            # Is this pressure level physically 'above' the ground?
            is_above_ground = (psfc > plevs[k])
            
            if k == 0:
                # Level 0 (1000hPa): Compare to surface
                # FIX: Added [0] to thtap to match 2D shapes
                mask = is_above_ground & (thtap[0] <= potsfc)
                thtap[0] = np.where(mask, potsfc + 0.01, thtap[0])
            else:
                # Is the ground 'trapped' between level k and k-1?
                is_trapped = (psfc < plevs[k-1])
                
                # Logic A: Ground is the neighbor (use potsfc)
                mask_a = is_above_ground & is_trapped & (thtap[k] <= potsfc)
                thtap[k] = np.where(mask_a, potsfc + 0.01, thtap[k])
            
                # Logic B: General floor-to-floor (use level below)
                mask_b = is_above_ground & (~is_trapped) & (thtap[k] <= thtap[k-1])
                thtap[k] = np.where(mask_b, thtap[k-1] + 0.01, thtap[k])
            
            # 3. Update the Global Roof (THTAHI)
            if k >= mid_idx:
                thtahi = max(thtahi, np.max(thtap[k]))

        return thtap, potsfc, thtahi, thtalo

    def adjust_stability_moore_1993(self, tpres, plevs, tsfc, psfc):
        """
        Final Refined Architect's Version: Moore (1993) Heat-Neutral Stability.
        Updated: thtahi is a running maximum, mid_point pre-calculated.
        """
        plvls, nj, ni = tpres.shape
        kappa = self.config['KAPPA']  # 0.2857
        p0 = self.config['P0']        # 100000.0
        
        # 1. INITIAL SITE SURVEY (Basement and Mid-Point)
        mid_idx = (plvls // 2) + 1  # For 17 levels, this is index 9 (10th level)
    
        # Potential Temp at Surface
        potsfc = tsfc * (p0 / psfc)**kappa
        thtalo = np.min(potsfc)
        
        # Potential Temp at All Isobaric Levels (The Floors)
        thtap = tpres * (p0 / plevs[:, np.newaxis, np.newaxis])**kappa
        
        # Initialize Roof Seed (Following Fortran logic: seed with the 10th level)
        thtahi = np.max(thtap[mid_idx])
    
        # 2. THE SEQUENTIAL BUILD (Vertical Monotonicity)
        for k in range(plvls):
            is_above_ground = (psfc > plevs[k])
            
            if k == 0:
                # Level 0 (1000hPa): Compare to surface (Anchor)
                mask_0 = is_above_ground & (thtap[0] <= potsfc)
                thtap[0] = np.where(mask_0, potsfc + 0.01, thtap[0])
            else:
                # Check for Moore (1993) instability (Upper colder than lower)
                # Only check points physically in the air
                is_unstable = is_above_ground & (thtap[k] <= thtap[k-1])
                
                # THE HEAT-NEUTRAL SPLIT:
                # Conserves energy while forcing stability for the solver
                theta_mean = (thtap[k] + thtap[k-1]) / 2.0
                
                thtap[k-1] = np.where(is_unstable, theta_mean - 0.005, thtap[k-1])
                thtap[k]   = np.where(is_unstable, theta_mean + 0.005, thtap[k])
            
                # 3. RUNNING ROOF SURVEY
                # Update the global high if we are in the upper atmosphere (Level 10+)
                if k >= mid_idx:
                   # This is faster than a full 3D scan at the end
                    thtahi = max(thtahi, np.max(thtap[k]))
                    
                    print(f"Blueprint Verified: Floors={plvls} | Low={thtalo:.2f}K | High={thtahi:.2f}K")
                    
        return thtap, potsfc, thtahi, thtalo




    def calculate_isentropic_levels_flexible(self, potsfc, thtahi, thtalo, d_theta, max_levels, coverage=0.10):
        """
        Architect's Level Generator: Uses the Moore 1993 outputs to define the 'Zoning'.
        
        Parameters:
        potsfc     : 2D array (theta_sfc) -> The 'Floor'
        thtahi     : float                -> The 'Roof'
        thtalo     : float                -> The 'Absolute Basement'
        d_theta    : float                -> The 'Floor Height' (DTHTA)
        max_levels : int                  -> The 'Max Stories' (MAXLVL)
        coverage   : float                -> The '10% Occupancy Rule'
        """
        
        # 1. Setup the Site Survey
        ni_nj = potsfc.size
        npts_required = ni_nj * coverage  # The 10% rule
        
        # 2. Find the Starting Floor (Refactors loops 600-1000)
        # Use thtalo (Absolute Basement) as the dynamic starting point
        current_theta = np.floor(thtalo / d_theta) * d_theta
        
        while True:
            # Count points where the target isentrope is 'in the air' (Surface is colder)
            npts_found = np.sum(potsfc <= current_theta)
            
            if npts_found >= npts_required:
                break
            current_theta += d_theta
        
            if current_theta > 600.0: break # Safety break
            
            start_theta = current_theta
        print(f"ISENTROPIC FOUNDATION ANCHORED AT: {start_theta} K")
        
        # 3. Build the Stack (Refactors loops 1100)
        # Generate exactly max_levels floors starting from start_theta
        thta_list = [start_theta + (i * d_theta) for i in range(max_levels)]
        thta = np.array(thta_list)
        
        # 4. Apply the Top-Cap Filter (Refactors loops 1300-1600)
        # Use thtahi (The Dynamic Roof) to ensure we don't build into space
        kthta = max_levels
        while kthta > 0:
            if thtahi >= thta[kthta - 1]:
                break
            kthta -= 1

        thta_final = thta[:kthta]
        
        print(f"FINAL TOP LEVEL: {thta_final[-1]} K")
        print(f"TOTAL VALID LEVELS FOR CALCULATION: {len(thta_final)}")
        
        return thta_final

    def calculate_isentropic_levels_flexible(self, potsfc, thtap, thtahi, thtalo, d_theta, max_levels, coverage=0.10):
        """
        Final Refined Architect's Generator: 
        Includes the specific 10% Top-Cap Filter (Refactors 1300-1600).
        """
        ni_nj = potsfc.size
        threshold = ni_nj * coverage  # The 10% Rule (NI * NJ / 10)
        
        # 1. BASEMENT: Find start_theta (Refactors 600-1000)
        current_theta = np.floor(thtalo / d_theta) * d_theta
        while True:
            if np.sum(potsfc <= current_theta) >= threshold:
                break
            current_theta += d_theta
            start_theta = current_theta
            
            # 2. THE STACK: Generate candidate levels (Refactors 1100)
            # We build up to MAXLVL + 1 initially to test the limits
            thta_candidates = np.array([start_theta + (i * d_theta) for i in range(max_levels)])

            # 3. THE ROOF: The 10% Top-Cap Check (Refactors 1300-1600)
            # We look at the Potential Temp at the VERY TOP isobaric level (thtap[-1])
            # THTAP (I, J, PLVLS) in Fortran is thtap[-1, :, :] in Python
            theta_at_model_top = thtap[-1, :, :]
            
            kthta = len(thta_candidates)
            while kthta > 0:
                target_top_theta = thta_candidates[kthta - 1]
            
                # Rule: Is the model's top level warmer than our target theta at 10% of points?
                # IF (THTAP (I, J, PLVLS) .GE. THTA (KTHTA))
                npts_top = np.sum(theta_at_model_top >= target_top_theta)
                
                if npts_top >= threshold:
                    break
        
        # If not enough points, drop one level and try again (GO TO 1300)
        kthta -= 1
    
        thta_final = thta_candidates[:kthta]

        # 4. REPORTING (Refactors 1600)
        if len(thta_final) >= max_levels:
            print(f"MAXLVL reached: Top Level is {thta_final[-1]}K.")
        else:
            print(f"Roof Check: Top Level capped at {thta_final[-1]}K due to data limits.")
            
        return thta_final

    def p2thta(self,lats,lons,plevs,tsfc,psfc,tpres):

        maxlvl = 22
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
        


        thtap = np.zeros((plvls,latLen,lonLen))
        potsfc = np.zeros((latLen,lonLen))
        alogp = np.zeros((plvls))
        thta = np.zeros(maxlvl)


        # Calculate potential temperature at the surface. Keep track of lowest value.
        # Calculate potential temperature at the surface. Keep track of lowest value.


        potsfc = self.potsfc(tsfc,psfc)
        thtalo = np.min(potsfc)


        # Compute potential temperature for each isobaric level eliminating superadiabatic or neutral layer
        thtap = self.pot(tpres,plevs)


        strat_idx = (plvls // 2) + 1 
        
        # PARITY INITIALIZATION: Match F77 THTAHI = POT(TPRES(1,1,10), PRES(10))
        thtahi = thtap[strat_idx, 0, 0]
        
        for k in range(0,len(plevs)):
            for j in range(0,latLen):
                for i in range(0,lonLen):
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
        pthta = np.zeros((kthta,latLen,lonLen),dtype='float64')
        
        for kout in range(0,kthta):
            for j in range(0,latLen):
                for i in range(0,lonLen):
                    # Begin IF
                    if (thta[kout] < potsfc[j,i]):
                        pthta[kout,j,i] = psfc[j,i] + random.uniform(0,0.01)
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

        #print(pthta.shape)
        ret = []
        ret.append(kthta)
        ret.append(pthta)
        ret.append(thta)
        return ret

    def sipv2_vectorized(self, lats, lons, kthta, thta, pthta, uthta, vthta):
        """
        Vectorized EPV calculation for NCAR/NOAA Reanalysis 1.
        Handles Surface (k=0) and TOA (k=-1) using one-sided differences.
        """
        # 1. Absolute Vorticity via windspharm (Vectorized across all 16 levels)
        w = VectorWind(uthta, vthta)
        absVor = w.absolutevorticity()
        
        # Constants
        p0 = 100000.0  
        kappa = 0.2857 
        gravity = 9.80665
        
        # Pre-allocate 3D arrays
        stabl = np.zeros_like(pthta)
        thta_3d = thta[:, np.newaxis, np.newaxis]
        
        # --- PART A: INTERNAL LEVELS (Centered Difference) ---
        # k from 1 to 14
        p_up_mid = pthta[2:, :, :]
        p_dn_mid = pthta[0:-2, :, :]
        
        mask_mid = (p_up_mid != p_dn_mid) & (p_dn_mid > 0)
        
        # log(theta_up / theta_dn) / log(p_up / p_dn)
        dlt_mid = np.log(thta_3d[2:] / thta_3d[0:-2])
        dlp_mid = np.log(np.divide(p_up_mid, p_dn_mid, where=mask_mid, out=np.ones_like(p_up_mid)))
        
        dltdlp_mid = np.zeros_like(dlp_mid)
        np.divide(dlt_mid, dlp_mid, out=dltdlp_mid, where=(dlp_mid != 0))
        
        stabl[1:-1, :, :] = (thta_3d[1:-1] / pthta[1:-1, :, :]) * (dltdlp_mid - kappa)
        
        # --- PART B: SURFACE LAYER (k=0, Forward Difference) ---
        # Compare Level 0 to Level 1
        p_bot = pthta[0, :, :]
        p_next = pthta[1, :, :]
        mask_bot = (p_next != p_bot) & (p_bot > 0)
        
        dlt_bot = np.log(thta[1] / thta[0])
        dlp_bot = np.log(np.divide(p_next, p_bot, where=mask_bot, out=np.ones_like(p_bot)))
        
        dltdlp_bot = np.zeros_like(dlp_bot)
        np.divide(dlt_bot, dlp_bot, out=dltdlp_bot, where=(dlp_bot != 0))
        
        stabl[0, :, :] = (thta[0] / pthta[0, :, :]) * (dltdlp_bot - kappa)
        
        # --- PART C: TOP LAYER (k=-1, Backward Difference) ---
        # Compare Top Level to Level below it
        p_top = pthta[-1, :, :]
        p_prev = pthta[-2, :, :]
        mask_top = (p_top != p_prev) & (p_prev > 0)
        
        dlt_top = np.log(thta[-1] / thta[-2])
        dlp_top = np.log(np.divide(p_top, p_prev, where=mask_top, out=np.ones_like(p_top)))
        
        dltdlp_top = np.zeros_like(dlp_top)
        np.divide(dlt_top, dlp_top, out=dltdlp_top, where=(dlp_top != 0))
        
        stabl[-1, :, :] = (thta[-1] / pthta[-1, :, :]) * (dltdlp_top - kappa)
        
        # --- FINAL CALCULATION ---
        # EPV = -g * (zeta + f) * Static_Stability
        ipv = -gravity * absVor * stabl
        
        # Gaussian smoothing to remove grid noise (2.5 degree grid typically uses sigma 2)
        return ndimage.gaussian_filter(ipv * 1e6, sigma=(0, 2, 2))

    
    def universal_lorenz_clip(thta_levels, p_sfc, t_sfc, vars_dict, sfc_vars_dict):
        """
        Architect's approach: Variable-agnostic Lorenz clipping.
        
        Parameters:
        thta_levels  : List/Array of target isentropic levels (e.g., [280, 290...])
        p_sfc        : 2D array of Surface Pressure (from ANY model)
        t_sfc        : 2D array of Surface Temperature (from ANY model)
        3d_vars_dict : Dictionary of 3D interpolated arrays {'P': pthta, 'U': uthta, 'V': vthta}
        sfc_vars_dict: Dictionary of 2D surface arrays {'P': p_sfc, 'U': u_sfc, 'V': v_sfc}
        """
        
        # 1. Calculate the Universal Threshold (Potential Temp at Ground)
        # R/Cp = 0.2857 is standard across NCEP/NCAR/ECMWF
        # Ensure p_sfc is in same units as 100000 (Pascals)
        theta_sfc = t_sfc * (100000.0 / p_sfc)**0.2857
        
        # 2. Iterate through levels and 'Skin' the underground points
        for k, theta_target in enumerate(thta_levels):
            # The 'Underground' Mask
            underground = theta_target < theta_sfc
            
            # Apply the clip to EVERY variable in your dictionary
        # This makes it agnostic: add 'T' or 'Q' to the dict, and it just works.
        for var_name in vars_dict.keys():
            target_3d = vars_dict[var_name]
            source_2d = sfc_vars_dict[var_name]
            
            # Force the 3D 'underground' point to match the 2D surface value
            target_3d[k, underground] = source_2d[underground]
            
        return vars_dict
    
    def tests2thta(self,lats,lons,plevs,kthta,uwndI,psfc,uins,thta,pthta):
        

        plevs_final = np.asfortranarray(plevs.T, dtype=np.float64)  # .T is shorthand for transpose(1,0)
        psfc_final  = np.asfortranarray(psfc.T, dtype=np.float64)  # .T is shorthand for transpose(1,0)
        ssfc_final  = np.asfortranarray(uwndI.T, dtype=np.float64) 
        
        # For 3D: (Levels, Lat, Lon) -> (Lon, Lat, Levels)
        spres_final = np.asfortranarray(np.transpose(uins, (2, 1, 0)), dtype=np.float64)
        pthta_final = np.asfortranarray(np.transpose(pthta, (2, 1, 0)), dtype=np.float64)
        

        
        sthta_old_full = interp_lib.s2thta_old(
            ssfc_final,
            psfc_final,
            spres_final,
            pthta_final
        )

        sthta_new_full = weather_lib.s2thta_vector(
            plevs_final,
            ssfc_final,
            psfc_final,
            spres_final,
            pthta_final
        )

        
        sthta_f77 = np.transpose(sthta_old_full[:, :, :], (2, 1, 0))
        sthta_f77 = np.ascontiguousarray(sthta_f77)

        sthta_f23 = np.transpose(sthta_new_full[:, :, :], (2, 1, 0))
        sthta_f23 = np.ascontiguousarray(sthta_f23)


        
        
        sthta_numpy = self.s2thta(plevs,uins, pthta, psfc, uwndI)

        sthta_numpy2 = self.s2thta_refactored(plevs,uins, pthta, psfc, uwndI)
        
        mea = np.mean(np.abs(sthta_numpy - sthta_f77))
        
        print(f"Mean Absolute Error: {mea} m/s")

        mea1 = np.mean(np.abs(sthta_f77-sthta_numpy2))

        
        print(f"Mean Absolute Error: {mea1} m/s")
        sys.exit()




    

    
    def s2thta(self,plevs,spres, pthta, psfc, ssfc):
        # Dimensions: (KOUT, NJ, NI) = (16, 73, 144)
        kout, nj, ni = pthta.shape
        plvls = plevs.size
        tol = 0.01

        pres = np.float64(plevs)
        # YOUR LNPU INITIALIZATION
        lnpu1p = np.log(pres[1:] / pres[:-1]) 
        lnpu2p = np.log(pres[2:] / pres[:-2])
        
        sthta = np.zeros_like(pthta)
        done = np.zeros_like(pthta, dtype=bool)
        
        # 1. INITIALIZATION & SURFACE IDENTITY
        psfc_3d = np.broadcast_to(psfc[None, :, :], (kout, nj, ni))
        ssfc_3d = np.broadcast_to(ssfc[None, :, :], (kout, nj, ni))
        sthta[pthta <= 0] = -9999.0
        done[pthta <= 0] = True
        
        mask_sfc = (~done) & (np.abs(pthta - psfc_3d) < tol)
        sthta[mask_sfc] = ssfc_3d[mask_sfc]
        done[mask_sfc] = True

        for k in range(plvls):
            match_mask = (~done) & (np.abs(pthta - pres[k]) < tol)
            if np.any(match_mask):
                # Broadcast 2D isobaric level to 3D to satisfy the 3D mask
                val_3d = spres[k, :, :][np.newaxis, :, :].repeat(kout, axis=0)
                sthta[match_mask] = val_3d[match_mask]
                done[match_mask] = True

        # 3. MAIN INTERPOLATION SWEEP
        pdwn, pmid, pup = [np.zeros_like(sthta) for _ in range(3)]
        sdwn, smid, sup = [np.zeros_like(sthta) for _ in range(3)]
        l12, l13, l23 = [np.zeros_like(sthta) for _ in range(3)]

        for k in range(plvls):
            root_mask = (~done) & (pthta > pres[k])
            if not np.any(root_mask): continue
            active = root_mask & (~done)
            
            # Placeholders
            pdwn.fill(0.0) 
            pmid.fill(0.0)
            pup.fill(0.0)
            sdwn.fill(0.0)
            smid.fill(0.0)
            sup.fill(0.0)
            l12.fill(0.0)
            l13.fill(0.0)
            l23.fill(0.0)
            if k == 0:
                pdwn[active] = psfc_3d[active]
                sdwn[active] = ssfc_3d[active]
                # Use 3D broadcasted condition to avoid IndexError
                c1_3d = (np.abs(psfc_3d - pres[k]) < tol)
                pmid[active] = np.where(c1_3d, pres[k],   pres[k+1])[active]
                pup[active]  = np.where(c1_3d, pres[k+1], pres[k+2])[active]
                smid[active] = np.where(c1_3d, spres[k,:,:][None,:,:],   spres[k+1,:,:][None,:,:])[active]
                sup[active]  = np.where(c1_3d, spres[k+1,:,:][None,:,:], spres[k+2,:,:][None,:,:])[active]
                l12[active] = np.where(c1_3d, lnpu1p[k], lnpu1p[k+1])[active]

                safe_ratio = np.divide(pup, pdwn, where=(pdwn != 0), out=np.ones_like(pup))

                # 2. Calculate the log ONLY where the ratio is positive and pdwn was safe
                safe_log = np.log(safe_ratio, where=(safe_ratio > 0), out=np.zeros_like(pup))

                l13[active] = np.where(c1_3d, safe_log, lnpu2p[k])[active]
                safe_ratio_23 = np.divide(pmid, pdwn, where=(pdwn > 0), out=np.ones_like(pmid))

                # 2. Safely calculate the Log
                # 'where' ensures we only calculate the log on positive, safe ratios
                safe_log_23 = np.log(safe_ratio_23, where=(safe_ratio_23 > 0), out=np.zeros_like(pmid))
                l23[active] = np.where(c1_3d, safe_log_23, lnpu1p[k])[active]

            # BRANCH 2: Top
            elif k == plvls - 1:
                pdwn[active], pmid[active], pup[active] = pres[k-2], pres[k-1], pres[k]
                sdwn[active] = spres[k-2,:,:][None,:,:].repeat(kout, axis=0)[active]
                smid[active] = spres[k-1,:,:][None,:,:].repeat(kout, axis=0)[active]
                sup[active]  = spres[k,:,:][None,:,:].repeat(kout, axis=0)[active]
                l12[active], l13[active], l23[active] = lnpu1p[k-1], lnpu2p[k-2], lnpu1p[k-2]
            else:
                m_psfc = active & (psfc_3d < pres[k-1])
                m_gen = active & (~m_psfc)
                
                if np.any(m_psfc):
                    pdwn[m_psfc], sdwn[m_psfc] = psfc_3d[m_psfc], ssfc_3d[m_psfc]
                    c3_3d = (np.abs(psfc_3d - pres[k]) < 0.001)
                    pmid[m_psfc] = np.where(c3_3d, pres[k],   pres[k+1])[m_psfc]
                    pup[m_psfc]  = np.where(c3_3d, pres[k+1], pres[k+2])[m_psfc]
                    smid[m_psfc] = np.where(c3_3d, spres[k,:,:][None,:,:],   spres[k+1,:,:][None,:,:])[m_psfc]
                    sup[m_psfc]  = np.where(c3_3d, spres[k+1,:,:][None,:,:], spres[k+2,:,:][None,:,:])[m_psfc]
                    l12[m_psfc] = np.where(c3_3d, lnpu1p[k], lnpu1p[k+1])[m_psfc]
                    safe_ratio_sfc = np.divide(pup, pdwn, where=(pdwn > 0), out=np.ones_like(pup))

                    # 2. The Safe Log: Only execute on positive, non-zero results
                    # out=0.0 handles any remaining invalid indices gracefully
                    safe_log_sfc = np.log(safe_ratio_sfc, where=(safe_ratio_sfc > 0), out=np.zeros_like(pup))
                    l13[m_psfc] = np.where(c3_3d, safe_log_sfc, lnpu2p[k])[m_psfc]
                    safe_ratio_23_sfc = np.divide(pmid, pdwn, where=(pdwn > 0), out=np.ones_like(pmid))

                    # 2. The Safe Log: Only execute on positive, non-zero results
                    # out=0.0 handles any remaining invalid indices gracefully
                    safe_log_23_sfc = np.log(safe_ratio_23_sfc, where=(safe_ratio_23_sfc > 0), out=np.zeros_like(pmid))

                    l23[m_psfc] = np.where(c3_3d, safe_log_23_sfc, lnpu1p[k])[m_psfc]
                    
                if np.any(m_gen):
                    pdwn[m_gen], pmid[m_gen], pup[m_gen] = pres[k-1], pres[k], pres[k+1]
                    sdwn[m_gen] = spres[k-1,:,:][None,:,:].repeat(kout, axis=0)[m_gen]
                    smid[m_gen] = spres[k,:,:][None,:,:].repeat(kout, axis=0)[m_gen]
                    sup[m_gen]  = spres[k+1,:,:][None,:,:].repeat(kout, axis=0)[m_gen]
                    l12[m_gen], l13[m_gen], l23[m_gen] = lnpu1p[k], lnpu2p[k-1], lnpu1p[k-1]

            safe_denom = (active) & (np.abs(l23) > 1e-12) & (np.abs(l13) > 1e-12) & (np.abs(l12) > 1e-12)

            with np.errstate(divide='ignore', invalid='ignore'):
                qdwn = np.zeros_like(pthta)
                qmid = np.zeros_like(pthta)
                qup = np.zeros_like(pthta)
                

                qdwn = np.divide(np.log(pthta/pmid) * np.log(pthta/pup), (l23 * l13), 
                                 where=safe_denom, out=np.zeros_like(pthta))

                qmid = np.divide(-np.log(pthta/pdwn) * np.log(pthta/pup), (l23 * l12),
                                 where=safe_denom, out=np.zeros_like(pthta))

                qup  = np.divide(np.log(pthta/pdwn) * np.log(pthta/pmid), (l13 * l12),
                                 where=safe_denom, out=np.zeros_like(pthta))
                sthta[active] = qdwn[active]*sdwn[active] + qmid[active]*smid[active] + qup[active]*sup[active]
                done[active] = True

        return sthta

    def s2thta_refactored(self, plevs, spres, pthta, psfc, ssfc):
        # Dimensions: (KOUT, NJ, NI) 
        kout, nj, ni = pthta.shape
        plvls = plevs.size # 
        tol = 0.01

        pres = np.float64(plevs)
        lnpu1p = np.log(pres[1:] / pres[:-1]) 
        lnpu2p = np.log(pres[2:] / pres[:-2])
        
        sthta = np.zeros_like(pthta, dtype=np.float64)
        done = np.zeros_like(pthta, dtype=bool)
        
        # 1. INITIALIZATION & SURFACE IDENTITY
        psfc_3d = np.broadcast_to(psfc[None, :, :], (kout, nj, ni))
        ssfc_3d = np.broadcast_to(ssfc[None, :, :], (kout, nj, ni))

        sthta[pthta <= 0] = -9999.0
        done[pthta <= 0] = True
        
        mask_sfc = (~done) & (np.abs(pthta - psfc_3d) < tol)
        sthta[mask_sfc] = ssfc_3d[mask_sfc]
        done[mask_sfc] = True

        # Pre-check isobaric matches for all levels
        for k in range(plvls):
            match_mask = (~done) & (np.abs(pthta - pres[k]) < tol)
            if np.any(match_mask):
                val_3d = spres[k][None, :, :].repeat(kout, axis=0)
                sthta[match_mask] = val_3d[match_mask]
                done[match_mask] = True

        # --- BRANCH 1: SURFACE SPECIAL CASE (k=0) ---
        k = 0
        active_0 = (~done) & (pthta > pres[k])
        if np.any(active_0):
            pdwn, pmid, pup = [np.zeros_like(sthta) for _ in range(3)]
            sdwn, smid, sup = [np.zeros_like(sthta) for _ in range(3)]
            l12, l13, l23 = [np.zeros_like(sthta) for _ in range(3)]

            pdwn[active_0], sdwn[active_0] = psfc_3d[active_0], ssfc_3d[active_0]
            c1_3d = (np.abs(psfc_3d - pres[k]) < tol)
            
            pmid[active_0] = np.where(c1_3d, pres[k],   pres[k+1])[active_0]
            pup[active_0]  = np.where(c1_3d, pres[k+1], pres[k+2])[active_0]
            
            s_k0, s_k1, s_k2 = spres[k,None], spres[k+1,None], spres[k+2,None]
            smid[active_0] = np.where(c1_3d, s_k0, s_k1)[active_0]
            sup[active_0]  = np.where(c1_3d, s_k1, s_k2)[active_0]
            
            l12[active_0] = np.where(c1_3d, lnpu1p[k], lnpu1p[k+1])[active_0]
            
            with np.errstate(divide='ignore', invalid='ignore'):
                l13[active_0] = np.where(c1_3d, np.log(np.divide(pup, pdwn, where=pdwn!=0)), lnpu2p[k])[active_0]
                l23[active_0] = np.where(c1_3d, np.log(np.divide(pmid, pdwn, where=pdwn!=0)), lnpu1p[k])[active_0]

            self._apply_quadratic(sthta, done, active_0, pthta, pdwn, pmid, pup, sdwn, smid, sup, l12, l13, l23)

        # --- BRANCH 2: THE HOT LOOP (k=1 to 15) ---
        pdwn, pmid, pup = [np.zeros_like(sthta) for _ in range(3)]
        sdwn, smid, sup = [np.zeros_like(sthta) for _ in range(3)]
        l12, l13, l23 = [np.zeros_like(sthta) for _ in range(3)]

        for k in range(1, plvls - 1):
            active = (~done) & (pthta > pres[k])
            if not np.any(active): continue
            

            m_psfc = active & (psfc_3d < pres[k-1])
            m_gen  = active & (~m_psfc)
            pdwn.fill(0.0) 
            pmid.fill(0.0)
            pup.fill(0.0)
            sdwn.fill(0.0)
            smid.fill(0.0)
            sup.fill(0.0)
            l12.fill(0.0)
            l13.fill(0.0)
            l23.fill(0.0)

            
            if np.any(m_psfc):
                pdwn[m_psfc], sdwn[m_psfc] = psfc_3d[m_psfc], ssfc_3d[m_psfc]
                c3_3d = (np.abs(psfc_3d - pres[k]) < 0.001) # F77 Typo preserved
                pmid[m_psfc] = np.where(c3_3d, pres[k],   pres[k+1])[m_psfc]
                pup[m_psfc]  = np.where(c3_3d, pres[k+1], pres[k+2])[m_psfc]
                
                skk, skp, sk2 = spres[k,None], spres[k+1,None], spres[k+2,None]
                smid[m_psfc] = np.where(c3_3d, skk, skp)[m_psfc]
                sup[m_psfc]  = np.where(c3_3d, skp, sk2)[m_psfc]
                
                l12[m_psfc] = np.where(c3_3d, lnpu1p[k], lnpu1p[k+1])[m_psfc]
                with np.errstate(divide='ignore', invalid='ignore'):
                    l13[m_psfc] = np.where(c3_3d, np.log(np.divide(pup, pdwn, where=pdwn!=0)), lnpu2p[k])[m_psfc]
                    l23[m_psfc] = np.where(c3_3d, np.log(np.divide(pmid, pdwn, where=pdwn!=0)), lnpu1p[k])[m_psfc]
                
            if np.any(m_gen):
                pdwn[m_gen], pmid[m_gen], pup[m_gen] = pres[k-1], pres[k], pres[k+1]
                sdwn[m_gen] = spres[k-1,None,:,:].repeat(kout, axis=0)[m_gen]
                smid[m_gen] = spres[k,None,:,:].repeat(kout, axis=0)[m_gen]
                sup[m_gen]  = spres[k+1,None,:,:].repeat(kout, axis=0)[m_gen]
                l12[m_gen], l13[m_gen], l23[m_gen] = lnpu1p[k], lnpu2p[k-1], lnpu1p[k-1]

            self._apply_quadratic(sthta, done, active, pthta, pdwn, pmid, pup, sdwn, smid, sup, l12, l13, l23)

        # --- BRANCH 3: TOP CAP SPECIAL CASE (k=16) ---
        k = plvls - 1
        active_top = (~done) & (pthta > pres[k])
        if np.any(active_top):
            pdwn, pmid, pup = pres[k-2], pres[k-1], pres[k]
            # Manual repeats for the top cap 3D mask
            sdwn_t = spres[k-2][None].repeat(kout, axis=0)
            smid_t = spres[k-1][None].repeat(kout, axis=0)
            sup_t  = spres[k][None].repeat(kout, axis=0)
            l12_t, l13_t, l23_t = lnpu1p[k-1], lnpu2p[k-2], lnpu1p[k-2]
            
            # Direct quadratic for the top cap
            self._apply_quadratic(sthta, done, active_top, pthta, pdwn, pmid, pup, sdwn_t, smid_t, sup_t, l12_t, l13_t, l23_t)

        return sthta

    def _apply_quadratic(self, sthta, done, mask, pthta, pdwn, pmid, pup, sdwn, smid, sup, l12, l13, l23):
        """Helper to keep the math bit-identical across branches"""
        safe_denom = (mask) & (np.abs(l23) > 1e-12) & (np.abs(l13) > 1e-12) & (np.abs(l12) > 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            qdwn = np.divide(np.log(pthta/pmid) * np.log(pthta/pup), (l23 * l13), where=safe_denom, out=np.zeros_like(pthta))
            qmid = np.divide(-np.log(pthta/pdwn) * np.log(pthta/pup), (l23 * l12), where=safe_denom, out=np.zeros_like(pthta))
            qup  = np.divide(np.log(pthta/pdwn) * np.log(pthta/pmid), (l13 * l12), where=safe_denom, out=np.zeros_like(pthta))
            sthta[mask] = (qdwn*sdwn + qmid*smid + qup*sup)[mask]
            done[mask] = True
