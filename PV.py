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
import traceback

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


        ef ddx(self, s, lat, lon):
    """
    Calculates the zonal derivative (ds/dx) for a SW-start grid.
    Conforms to Brill's "East minus West" requirement.
    """
    # 1. Handle Missing Data: Convert -999.99 to NaNs for 2026 standards
    s = np.where(s <= -999.99, np.nan, s)
    lat_len, lon_len = s.shape
    dsdx = np.full((lat_len, lon_len), np.nan)

    # 2. Calculate Physical Distance (dx)
    # R * cos(lat) * dlon handles the convergence of meridians at the poles
    rearth = 6371221.3
    dlon_rad = np.abs(np.radians(lon[1] - lon[0]))
    
    # Create a latitude-dependent dx (column vector for broadcasting)
    # This ensures accuracy at high latitudes vs the Equator
    dx = (rearth * np.cos(np.radians(lat)) * dlon_rad).reshape(-1, 1)

    # 3. Create Valid Data Mask
    valid = ~np.isnan(s)


# --- INTERIOR: Centered Differences (East minus West) ---
    # s[:, 2:] is East, s[:, :-2] is West. Brill Requirement: (East - West)
    has_both = valid[:, 2:] & valid[:, :-2] & valid[:, 1:-1]
    
    # Calculate gradient where neighbors exist
    dsdx[:, 1:-1][has_both] = (s[:, 2:][has_both] - s[:, :-2][has_both]) / (2.0 * dx[has_both[:,0]])

    # --- CYCLIC BOUNDARY: Global Wrap-around Handling ---
    # Check if the grid spans 360 degrees
    is_cyclic = np.allclose((lon[-1] - lon[0]) % 360, 360 - (lon[1] - lon[0]), atol=1e-2)

    if is_cyclic:
        # Western Boundary (Index 0): (Point East - Point West [wrap])
        has_wrap_w = valid[:, 0] & valid[:, 1] & valid[:, -1]
        dsdx[:, 0][has_wrap_w] = (s[:, 1][has_wrap_w] - s[:, -1][has_wrap_w]) / (2.0 * dx.flatten()[has_wrap_w[:,0]])
    else:
        # Standard Boundaries: Fallback to One-Sided Differences
        # West Edge (Forward)
        has_west = valid[:, 0] & valid[:, 1]
        dsdx[:, 0][has_west] = (s[:, 1][has_west] - s[:, 0][has_west]) / dx.flatten()[has_west]
        
        # East Edge (Backward)
        has_east = valid[:, -1] & valid[:, -2]
        dsdx[:, -1][has_east] = (s[:, -1][has_east] - s[:, -2][has_east]) / dx.flatten()[has_east]

    # Return result with original missing value flag if desired
    return np.where(np.isnan(dsdx), -999.99, dsdx


def ddx_fixed(self, s, lat, lon):
    """
    Calculates zonal derivative (ds/dx) with a polar cap to prevent spikes.
    Follows Keith Brill's East-minus-West requirement for SW-start grids.
    """
    # 1. Handle Missing Data (Standard 2026 practice)
    s = np.where(s <= -999.99, np.nan, s)
    lat_len, lon_len = s.shape
    dsdx = np.full((lat_len, lon_len), np.nan)

    # 2. Calculate Physical Distance (dx) with Polar Stability
    rearth = 6371221.3
    dlon_rad = np.abs(np.radians(lon[1] - lon[0]))
    
    # Apply Latitude Cap: Prevents division by near-zero at poles
    # We cap at 80 degrees; this mimics the stability of spectral models
    lat_rad = np.radians(lat)
    safe_cos = np.cos(lat_rad)
    cos_cap = np.cos(np.radians(80.0))
    safe_cos = np.maximum(safe_cos, cos_cap)
# Reshape for broadcasting across longitudes
    dx = (rearth * safe_cos * dlon_rad).reshape(-1, 1)

    valid = ~np.isnan(s)

    # 3. INTERIOR: Centered Differences (East minus West)
    # s[:, 2:] is East, s[:, :-2] is West. Brill Requirement: (East - West)
    has_both = valid[:, 2:] & valid[:, :-2] & valid[:, 1:-1]
    dsdx[:, 1:-1][has_both] = (s[:, 2:][has_both] - s[:, :-2][has_both]) / (2.0 * dx[1:-1])

    # 4. CYCLIC BOUNDARY: Global Wrap-around
    # Use 1e-2 tolerance for floating point longitude comparisons
    is_cyclic = np.allclose((lon[-1] - lon[0]) % 360, 360 - (lon[1] - lon[0]), atol=1e-2)

    if is_cyclic:
        # Western Edge (Index 0): (Point East - Point West [wrap-around])
        has_wrap_w = valid[:, 0] & valid[:, 1] & valid[:, -1]
        dsdx[:, 0][has_wrap_w] = (s[:, 1][has_wrap_w] - s[:, -1][has_wrap_w]) / (2.0 * dx.flatten())
        
        # Eastern Edge (Index -1): (Point East [wrap-around] - Point West)
        has_wrap_e = valid[:, -1] & valid[:, -2] & valid[:, 0]
        dsdx[:, -1][has_wrap_e] = (s[:, 0][has_wrap_e] - s[:, -2][has_wrap_e]) / (2.0 * dx.flatten())
    else:
        # Standard Boundaries: Forward/Backward Difference
        has_west = valid[:, 0] & valid[:, 1]
        dsdx[:, 0][has_west] = (s[:, 1][has_west] - s[:, 0][has_west]) / dx.flatten()
        
        has_east = valid[:, -1] & valid[:, -2]
        dsdx[:, -1][has_east] = (s[:, -1][has_east] - s[:, -2][has_east]) / dx.flatten()

    # Final step: Return to original missing data flag
    return np.where(np.isnan(dsdx), -999.99, dsdx)

                    
    def ddy(self,s,lat,lon):
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

    def ddy_sw_start(self, s, lat, lon):
    # Standardize missing values to NaNs
    s = np.where(s == -999.99, np.nan, s)
    
    lat_len, lon_len = s.shape
    dsdy = np.full((lat_len, lon_len), np.nan)

    # Physical distance (assuming lat is in degrees)
    rearth = 6371221.3
    # Absolute difference ensures dj is positive regardless of lat order
    dj = np.abs(np.radians(lat[1] - lat[0]) * rearth)

    valid = ~np.isnan(s)

    # --- INTERIOR: Centered Differences (North - South) ---
    # In SW-start: s[2:] is North of s[:-2]
    has_both = valid[2:, :] & valid[:-2, :] & valid[1:-1, :]
    dsdy[1:-1, :][has_both] = (s[2:, :][has_both] - s[:-2, :][has_both]) / (2.0 * dj)

    # --- SOUTH BOUNDARY (Index 0): Forward Difference ---
    # North point (1) minus South point (0)
    has_south = valid[0, :] & valid[1, :]

    dsdy[0, :][has_south] = (s[1, :][has_south] - s[0, :][has_south]) / dj

    # --- NORTH BOUNDARY (Index -1): Backward Difference ---
    # North point (-1) minus point just South of it (-2)
    has_north = valid[-1, :] & valid[-2, :]
    dsdy[-1, :][has_north] = (s[-1, :][has_north] - s[-2, :][has_north]) / dj

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
    relv[1:-1, :] = dvdx[1:-1, :] - dudy[1:-1, :] + (u[1:-1, :] * tan_lat / rearth
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
    """
    Unified Vectorized Potential Temperature.
    Works for 3D (Level, Lat, Lon) or 2D (Lat, Lon) inputs.
    Eliminates the need for a separate potsfc function.
    """
    # 1. Pull constants from your centralized config (No more Magic Numbers)
    # self.phys contains RD, CP, and P0 loaded from your YAML/Config
    kappa = self.phys.KAPPA.m  # Rd/Cp
    p0 = self.phys.P0.m        # 100000.0
    
    # 2. Universal Dimension Handling
    # If pres is 1D (17,), expand to (17, 1, 1) to broadcast against (17, 73, 144)
    # If pres is 2D (73, 144), it will broadcast against 2D tmp naturally.
    if pres.ndim == 1:
        pres_resolved = pres[:, np.newaxis, np.newaxis]
    else:
        pres_resolved = pres
                                                     
    # 3. Core Poisson Equation
    # Vectorized: Processes the entire volume or surface in one SIMD sweep.
    # NaNs propagate naturally here.
    return tmp * (p0 / pres_resolved) ** kappa                                                 
  def pvonp_vectorized(self, ni, nj, lat, lon, pres, pres1, pres2, tmp, tmp1, tmp2, u, u1, u2, v, v1, v2):
    """
    Vectorized refactor of PV on Pressure levels.
    USP: Logarithmic Vertical Thermodynamics (ln T vs ln P).
    """
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
    
    def tests2thta(self,lats,lons,plevs,kthta,uwndI,psfc,uins,thta,pthta):

        uRef = self.s2thta(lats,lons,plevs,kthta,uwndI,psfc,uins,thta,pthta)

        u = self.s2thta_vectorized(uwndI,pthta, uins,psfc,plevs)

        latLen = len(lats)
        lonLen = len(lons)
        for k in range(0,1):
            for j in range(0,latLen):
                for i in range(0,lonLen):
                    print(uRef[k,j,i]-u[k,j,i])

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


        thtahi = thtap[9,0,0]
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

        print(pthta.shape)
        ret = []
        ret.append(kthta)
        ret.append(pthta)
        ret.append(thta)
        return ret

    def p2thta_vectorized(self, lats, lons, plevs, tsfc, psfc, tpres):
        # --- Constants (2026 Standards) ---
        kappa = 2.0 / 7.0
        p0 = 100000.0  # Reference pressure in Pa
        dthta = 10.0
        maxlvl = 17
        epsln = 0.001
        
        lat_len, lon_len = len(lats), len(lons)
        plvls = len(plevs)
        
        # 1. Potential Temperature at Surface & Levels (Vectorized)
        potsfc = tsfc * (p0 / psfc)**kappa
        thtalo = np.min(potsfc)
        
        # thtap: [level, lat, lon]
        thtap = tpres * (p0 / plevs[:, None, None])**kappa
        
        # 2. Stability Correction (Vectorized Running Maximum)
        # Ensures theta strictly increases with height (eliminates neutral/superadiabatic)
        for k in range(1, plvls):
            thtap[k] = np.maximum(thtap[k], thtap[k-1] + 0.01)
            thtap[k] = np.maximum(thtap[k], potsfc + 0.01)

            candidate_thtas = 200.0 + np.arange(100) * dthta
        valid_thtas = []
        for t_val in candidate_thtas:
            npts = np.sum(potsfc <= t_val)
            if npts >= (lat_len * lon_len) / 10.0:
                valid_thtas.append(t_val)
                if len(valid_thtas) >= maxlvl: break

        thta_levels = np.array(valid_thtas)
        kthta = len(thta_levels)

        
        indices = np.apply_along_axis(lambda c: np.searchsorted(c, thta_levels), 0, thtap)
        # Clip to ensure index-1 and index+0 are safe
        idx_up = np.clip(indices, 1, plvls - 1)
        idx_dn = idx_up - 1


        pot_up = np.take_along_axis(thtap, idx_up[None, ...], axis=0).squeeze(0)
        pot_dn = np.take_along_axis(thtap, idx_dn[None, ...], axis=0).squeeze(0)
        p_up = plevs[idx_up]
        p_dn = plevs[idx_dn]
        
        ln_pup = np.log(p_up)
        ln_pdn = np.log(p_dn)


        th_target = thta_levels[:, None, None] 
    

        t_up = pot_up * (p_up / p0)**kappa
        t_dn = pot_dn * (p_dn / p0)**kappa
    
        dltdlp = (np.log(t_up) - np.log(t_dn)) / (ln_pup - ln_pdn)
        interc = np.log(t_up) - dltdlp * ln_pup


        pln = ln_pdn + 0.5 * (ln_pup - ln_pdn)
    

        for _ in range(5):
            # f(pln) = ln(theta_target) - ln(theta_current)
            # where ln(theta) = ln(T) + kappa * (ln(p0) - ln(p))
            # and ln(T) = interc + dltdlp * pln
            f = np.log(th_target) - (interc + dltdlp * pln + kappa * (np.log(p0) - pln))
            
            # Derivative: f'(pln) = -(dltdlp - kappa)
            df = -(dltdlp - kappa)
        
            # Update guess: x = x - f/f'
            pln = pln - f / df
            pthta = np.exp(pln)


        sfc_mask = th_target < potsfc[None, :, :]
        top_mask = th_target > thtap[-1, :, :]
        pthta[sfc_mask] = psfc[None, :, :][sfc_mask]
        pthta[top_mask] = 1.0 # Top of model
    
        return thta_levels, pthta

    def s2thta(self,lats,lons,pres,kthta,ssfc,psfc,spres,thta,pthta):

        plvls = len(pres)
        latLen = len(lats)
        lonLen = len(lons)
        sthta = np.zeros((kthta,latLen,lonLen))
        lnpu1p = np.zeros((plvls-1))
        lnpu2p = np.zeros((plvls-2))

        
        np.seterr(all='warn')
        warnings.filterwarnings('error')


        lnpu1p[0:-1] = np.log(pres[1:-1]/pres[0:-2])
        lnpu2p[0:] = np.log(pres[2:]/pres[:-2])
        lnpu1p[-1] = np.log(pres[-1]/pres[-2])


        for kout in range(0,kthta):
            for j in range(0,latLen):
                for i in range(0,lonLen):
                    if(pthta[kout,j,i] <= 0.):
                        sthta[kout,j,i] = -999.99
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
                                    #print(kin)
                                    if (abs(psfc[j,i]-pres[kin]) < 0.001):
                                        pmid = pres[kin]
                                        pup = pres[kin+1]
                                        smid = spres[kin,j,i]
                                        sup = spres[kin+1,j,i]
                                        lnp1p2 = float(lnpu1p[kin])
                                        lnp1p3 = float(np.log(pup/pdwn))
                                        if (pmid == pdwn):
                                            pmid += 0.01
                                        try:
                                            lnp2p3 = float(np.log(pmid/pdwn))
                                        except Warning:
                                            print(lnp2p3)
                                            print(traceback.format_exc())
                                            #print(pmid,pup)
                                    else:
                                        pmid = pres[kin+1]
                                        pup = pres[kin+2]
                                        smid = spres[kin+1,j,i]
                                        sup = spres[kin+2,j,i]
                                        lnp1p2 = float(lnpu1p[kin+1])
                                        lnp1p3 = float(lnpu2p[kin])
                                        lnp2p3 = float(lnpu1p[kin])
                                        #print(pmid,pup,smid,sup)
                                elif (kin == plvls-1):
                                    pdwn = pres[kin-2]
                                    pmid = pres[kin-1]
                                    pup = pres[kin]
                                    sdwn = spres[kin-2,j,i]
                                    smid = spres[kin-1,j,i]
                                    sup = spres[kin,j,i]
                                    lnp1p2 = float(lnpu1p[kin-1])
                                    lnp1p3 = float(lnpu2p[kin-2])
                                    lnp2p3 = float(lnpu1p[kin-2])
                                elif (psfc[j,i] < pres[kin-1]):
                                    pdwn = psfc[j,i]
                                    sdwn = ssfc[j,i]
                                    if (abs(psfc[j,i] -pres[kin]) > 0.001):
                                        pmid = pres[kin]
                                        pup =  pres[kin+1]
                                        smid = spres[kin,j,i]
                                        sup = spres[kin+1,j,i]
                                        lnp1p2 = float(lnpu1p[kin])
                                        if (pup == pdwn):
                                            pup += 0.01
                                        lnp1p3 = float(np.log(pup/pdwn))
                                        lnp2p3 = float(np.log(pmid/pdwn))
                                    else:
                                        pmid = pres[kin+1]
                                        pup = pres[kin+2]
                                        smid = spres[kin+1,j,i]
                                        sup = spres[kin+2,j,i]
                                        lnp1p2 = float(lnpu1p[kin+1])
                                        lnp1p3 = float(lnpu2p[kin])
                                        lnp2p3 = float(lnpu1p[kin])
                                else:
                                    pdwn = pres[kin-1]

                                    pmid = pres[kin]
                                    pup =  pres[kin+1]
                                    sdwn = spres[kin-1,j,i]
                                    smid = spres[kin,j,i]
                                    sup = spres[kin+1,j,i]
                                    lnp1p2 = float(lnpu1p[kin])
                                    lnp1p3 = float(lnpu2p[kin-1])
                                    lnp2p3 = float(lnpu1p[kin-1])
                                looping = False
                            kin +=1
                        try:
                            qdwn = np.log(pthta[kout,j,i]/pmid)*np.log(pthta[kout,j,i]/pup)/lnp2p3/lnp1p3
                            qmid = -np.log(pthta[kout,j,i]/pdwn)*np.log(pthta[kout,j,i]/pup)/lnp2p3/lnp1p2
                            qup = np.log(pthta[kout,j,i]/pdwn)*np.log(pthta[kout,j,i]/pmid)/lnp1p3/lnp1p2
                            sthta[kout,j,i] = qdwn*sdwn + qmid*smid + qup *sup
                        except Warning:
                            print(pup,pmid,lnp1p3,lnp2p3)
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            print(exc_type, exc_tb.tb_lineno)
                            print(traceback.format_exc())

        return sthta

import numpy as np

def s2thta_vectorized(spres, pthta, ssfc, psfc, pres_levels):
    """
    Interpolates a 2D wind field (spres) to a 3D target pressure field (pthta).
    
    Inputs:
    - spres: 2D array (73, 144) 
    - pthta: 3D array (17, 73, 144) 
    - ssfc, psfc: 2D surface data (73, 144)
    - pres_levels: 1D pressure levels (17,)
    """
    # 1. Define sthta internally (MetPy Protocol)
    # Inherit shape and float64 precision from pthta
    sthta = np.full(pthta.shape, -9999.0, dtype=np.float64)
    nlev, nlat, nlon = pthta.shape

    # 2. Gatekeeper Mask
    valid = pthta > 0.0

    # 3. Surface Pressure Match (Condition 2)
    # Broadcast 2D surface values across all 17 levels automatically
    sfc_match = valid & (np.abs(pthta - psfc[None, :, :]) < 0.01)
    sthta[sfc_match] = np.broadcast_to(ssfc[None, :, :], pthta.shape)[sfc_match]

    # 4. Vertical Search (Condition 3 & 7)
    # Find where each of the 17 target pressures sits in the 1D list
    # Result is a 3D index map (17, 73, 144)
    idx_kin = np.searchsorted(pres_levels[::-1], pthta, side='right')
    idx_kin = (len(pres_levels) - 1) - idx_kin
    idx_kin = np.clip(idx_kin, 0, len(pres_levels) - 1)

    # 5. Exact Level Match (Condition 3)
    pres_at_kin = pres_levels[idx_kin]
    exact_match = valid & ~sfc_match & (np.abs(pthta - pres_at_kin) < 0.01)
    
    # Since spres is 2D, we broadcast it to 3D for the match
    spres_3d = np.broadcast_to(spres[None, :, :], pthta.shape)
    sthta[exact_match] = spres_3d[exact_match]

    # 6. General Interpolation Stencil
    interp_mask = valid & ~sfc_match & ~exact_match
    
    # Pressure Stencil (from 1D pres_levels)
    pmid = pres_levels[idx_kin]
    pdwn = pres_levels[np.clip(idx_kin - 1, 0, len(pres_levels)-1)]
    pup  = pres_levels[np.clip(idx_kin + 1, 0, len(pres_levels)-1)]

    # Boundary: Inject 2D Surface (Condition 4 & 5)
    near_sfc = interp_mask & (idx_kin == 0)
    pdwn[near_sfc] = np.broadcast_to(psfc[None, :, :], pdwn.shape)[near_sfc]
    # No need to change sdwn here because we use the broadcasted spres_3d below

    # 7. Lagrange 4-Point Math
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_pt, ln_pd, ln_pm, ln_pu = np.log(pthta), np.log(pdwn), np.log(pmid), np.log(pup)
        ln12, ln13, ln23 = (ln_pm - ln_pd), (ln_pu - ln_pd), (ln_pu - ln_pm)

        # Standard 4-point weights
        qdwn = ((ln_pt - ln_pm) * (ln_pt - ln_pu)) / (ln12 * ln13)
        qmid = -((ln_pt - ln_pd) * (ln_pt - ln_pu)) / (ln12 * ln23)
        qup  = ((ln_pt - ln_pd) * (ln_pt - ln_pm)) / (ln13 * ln23)

        # Since we are interpolating a static 2D field, 
        # SDWN, SMID, and SUP are all the same broadcasted value!
        # This simplifies the math significantly.
        sthta[interp_mask] = (spres_3d * (qdwn + qmid + qup))[interp_mask]

    return sthta


    def s2thtaref(self,lats,lons,pres,kthta,ssfc,psfc,spres,thta,pthta):

        plvls = len(pres)
        latLen = len(lats)
        lonLen = len(lons)
        sthta = np.zeros((kthta,latLen,lonLen))
        lnpu1p = np.zeros((plvls-1))
        lnpu2p = np.zeros((plvls-2))
        #spres = np.transpose(spres,(2,0,1))

        np.seterr(all='warn')
        warnings.filterwarnings('error')
        
        pdwn = np.zeros((kthta))

        pmid = np.zeros((kthta))
        pup = np.zeros((kthta))
        sdwn = np.zeros((kthta,latLen,lonLen))
        smid = np.zeros((kthta,latLen,lonLen))
        sup = np.zeros((kthta,latLen,lonLen))
        qdwn = np.zeros((kthta,latLen,lonLen))
        qmid = np.zeros((kthta,latLen,lonLen))
        qup = np.zeros((kthta,latLen,lonLen))
        lnpu1p[0:-1] = np.log(pres[1:-1]/pres[0:-2])

        lnpu2p[0:] = np.log(pres[2:]/pres[:-2])
    
        lnpu1p[-1] = float(np.log(pres[-1]/pres[-2]))

        lnp1p2 = np.zeros((kthta))
        lnp1p3 = np.zeros((kthta))
        lnp2p3 = np.zeros((kthta))
        isPthtaLess1 = abs(pthta[:,:,:]-psfc[None,:,:]) < 0.001
        sthta = np.where(isPthtaLess1,ssfc[None,:,:],sthta)
        #print(
        pdwn[1:-1] = pres[0:-3]

        pmid[1:-1] = pres[1:-2]

        pup[1:-1] =  pres[2:-1]

        sdwn[1:-1,:,:] = spres[0:-3,:,:]

        smid[1:-1,:,:] = spres[1:-2,:,:]

        sup[1:-1,:,:] = spres[2:-1,:,:]

        
        lnp1p2[1:-1] = lnpu1p[1:-1]

        lnp1p3[1:-1] = lnpu2p[0:-1]

        lnp2p3[1:-1] = lnpu1p[0:-2]

        #lower boundary 
        for j in range(0,latLen):
            for i in range(0,lonLen):
                pdwnsur = psfc[j,i]
                sdwnsur = ssfc[j,i]
                if (pthta[0,j,i] > pres[0]):
                    if (abs(psfc[j,i]-pres[0]) < 0.001):
                        pmidsur = pres[0] + random.uniform(0,0.01)
                        pupsur = pres[1]
                        smidsur = spres[0,j,i]
                        supsur = spres[1,j,i]
                        lnp1p2sur = float(lnpu1p[0])
                        lnp1p3sur = float(np.log(pupsur/pdwnsur))
                        try:
                            lnp2p3sur = float(np.log(pmidsur/pdwnsur))
                        except Warning:
                            print(traceback.format_exc())
                            #print(pmid,pup)
                    else:
                        pmidsur = pres[1]
                        pupsur = pres[2]
                        smidsur = spres[1,j,i]
                        supsur = spres[2,j,i]
                        lnp1p2sur = float(lnpu1p[1])
                        lnp1p3sur = float(lnpu2p[0])
                        lnp2p3sur = float(lnpu1p[0])
                #print(pmid,pup,smid,sup)
                qdwnsur = np.log(pthta[0,j,i]/pmidsur)*np.log(pthta[0,j,i]/pupsur)/lnp2p3sur/lnp1p3sur
                qmidsur = -np.log(pthta[0,j,i]/pdwnsur)*np.log(pthta[0,j,i]/pupsur)/lnp2p3sur/lnp1p2sur
                qupsur = np.log(pthta[0,j,i]/pdwnsur)*np.log(pthta[0,j,i]/pmidsur)/lnp1p3sur/lnp1p2sur
                sthta[0,j,i] = qdwnsur*sdwnsur + qmidsur*smidsur + qupsur *supsur        

        #top

        pdwn[-1] = pres[-3]

        sdwn[-1,:,:] = spres[-3,:,:]

        pmid[-1] = pres[-2]
        smid[-1,:,:] = spres[-2,:,:]

        pup[-1] = pres[-1]
        sup[-1,:,:] = spres[-1,:,:]

        lnp1p2[-1] = lnpu1p[-1]

        lnp1p3[-1] =   lnpu2p[-3]

        lnp2p3[-1] = lnpu1p[-1]

        qdwn[-1,:,:] = np.log(pthta[-1,:,:]/pmid[-1,None,None])*np.log(pthta[-1,:,:]/pup[-1,None,None])/lnp2p3[-1,None,None]/lnp1p3[-1,None,None]

        qmid[-1,:,:] = -np.log(pthta[-1,:,:]/pdwn[-1,None,None])*np.log(pthta[-1,:,:]/pup[-1,None,None])/lnp2p3[-1,None,None]/lnp1p2[-1,None,None]

        qup[-1,:,:] = np.log(pthta[-1,:,:]/pdwn[-1,None,None])*np.log(pthta[-1,:,:]/pmid[-1,None,None])/lnp1p3[-1,None,None]/lnp1p2[-1,None,None]
        sthta[-1,:,:] = qdwn[-1,:,:]*sdwn[-1,:,:] + qmid[-1,:,:]*smid[-1,:,:] + qup[-1,:,:] *sup[-1,:,:]


        
        qdwn[1:,:,:] = np.log(pthta[1:,:,:]/pmid[1:,None,None])*np.log(pthta[1:,:,:]/pup[1:,None,None])/lnp2p3[1:,None,None]/lnp1p3[1:,None,None]

        qmid[1:,:,:] = -np.log(pthta[1,:,:]/pdwn[1:,None,None])*np.log(pthta[1:,:,:]/pup[1:,None,None])/lnp2p3[1:,None,None]/lnp1p2[1:,None,None]

        qup[1:,:,:] = np.log(pthta[1:,:,:]/pdwn[1:,None,None])*np.log(pthta[1:,:,:]/pmid[1:,None,None])/lnp1p3[1:,None,None]/lnp1p2[1:,None,None]
        sthta[1:,:,:] = qdwn[1:,:,:]*sdwn[1:,:,:] + qmid[1:,:,:]*smid[1:,:,:] + qup[1:,:,:] *sup[1:,:,:]

        return sthta
     
