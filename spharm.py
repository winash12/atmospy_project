import numpy as np
import shtns
import sys

class Spharmt(object):
     """
     wrapper class for commonly used spectral transform operations in
     atmospheric models.  Provides an interface to shtns compatible
     with pyspharm (pyspharm.googlecode.com).
     """
     def __init__(self, nlons, nlats, ntrunc, rsphere, gridtype="gaussian"):
         """initialize
         nlons:  number of longitudes
         nlats:  number of latitudes"""

         self._shtns = shtns.sht(ntrunc, ntrunc, 1,
                                 shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)
  
         if gridtype == "gaussian":
             # self._shtns.set_grid(nlats, nlons,
             #         shtns.sht_gauss_fly | shtns.SHT_PHI_CONTIGUOUS, 1.e-10)
             self._shtns.set_grid(nlats, nlons,
                     shtns.sht_quick_init | shtns.SHT_PHI_CONTIGUOUS, 1.e-10)
         elif gridtype == "regular":
             self._shtns.set_grid(nlats, nlons,
                     shtns.sht_reg_dct | shtns.SHT_PHI_CONTIGUOUS, 1.e-10)
  
         self.lats = np.arcsin(self._shtns.cos_theta)
         self.lons = (2.*np.pi/nlons)*np.arange(nlons)
         self.nlons = nlons
         self.nlats = nlats
         self.ntrunc = ntrunc
         self.nlm = self._shtns.nlm
         self.degree = self._shtns.l
         self.lap = -self.degree*(self.degree+1.0).astype(np.complex)
         self.invlap = np.zeros(self.lap.shape, self.lap.dtype)
         self.invlap[1:] = 1./self.lap[1:]
         self.rsphere = rsphere
         self.lap = self.lap/rsphere**2
         self.invlap = self.invlap*rsphere**2
         print(self.invlap)
         
     def grdtospec(self, data):
         """compute spectral coefficients from gridded data"""
         return self._shtns.analys(data)
  
     def spectogrd(self, dataspec):
         """compute gridded data from spectral coefficients"""
         return self._shtns.synth(dataspec)
  
     def getuv(self, vrtspec, divspec):
         """compute wind vector from spectral coeffs of vorticity and divergence"""
         return self._shtns.synth((self.invlap/self.rsphere)*vrtspec, (self.invlap/self.rsphere)*divspec)
  
     def getvrtdivspec(self, u, v):
         """compute spectral coeffs of vorticity and divergence from wind vector"""
         vrtspec, divspec = self._shtns.analys(u, v)
         vrtspec = self.lap*self.rsphere*vrtspec
         divspec = self.lap*self.rsphere*divspec
         return vrtspec,divspec
  
     def getgrad(self, divspec):
         """compute gradient vector from spectral coeffs"""
         vrtspec = np.zeros(divspec.shape, dtype=np.complex)
         u, v = self._shtns.synth(vrtspec, divspec)
         return u/rsphere, v/rsphere


