#! /usr/bin/env python3
  
 #
 # "Non-linear barotropically unstable shallow water test case"
 #
 # Running the script should pop up a window with this image:
 #  https://i.imgur.com/CEzHJ0g.png
 #
  
import numpy as np
import matplotlib.pyplot as plt
import time
from spharm import Spharmt
import sys  
  
def main():
  
     # non-linear barotropically unstable shallow water test case
     # of Galewsky et al (2004, Tellus, 56A, 429-440).
     # "An initial-value problem for testing numerical models of the global
     # shallow-water equations" DOI: 10.1111/j.1600-0870.2004.00071.x
     # http://www-vortex.mcs.st-and.ac.uk/~rks/reprints/galewsky_etal_tellus_2004.pdf
  
     # requires matplotlib for plotting.
  
     # grid, time step info
     #nlons = 256                 # number of longitudes
     nlons = 144
     ntrunc = int(nlons/3)       # spectral truncation (for alias-free computations)
     #nlats = int(nlons/2)        # for gaussian grid.
     nlats = 73
     dt = 150                    # time step in seconds
     itmax = 6*int(86400/dt)     # integration length in days
  
     # parameters for test
     rsphere = 6.37122e6         # earth radius
     omega = 7.292e-5            # rotation rate
     grav = 9.80616              # gravity
     hbar = 10.e3                # resting depth
     umax = 80.                  # jet speed
     phi0 = np.pi/7.
     phi1 = 0.5*np.pi - phi0
     phi2 = 0.25*np.pi
     en = np.exp(-4.0/(phi1-phi0)**2)
     alpha = 1./3.
     beta = 1./15.
     hamp = 120.                 # amplitude of height perturbation to zonal jet
     efold = 3.*3600.            # efolding timescale at ntrunc for hyperdiffusion
     ndiss = 8                   # order for hyperdiffusion
  
     # setup up spherical harmonic instance, set lats/lons of grid
     x = Spharmt(nlons, nlats, ntrunc, rsphere, gridtype="gaussian")
     lons, lats = np.meshgrid(x.lons, x.lats)
     f = 2.*omega*np.sin(lats)   # coriolis
  
     # zonal jet.
     vg = np.zeros((nlats, nlons), np.float)

     u1 = (umax/en)*np.exp(1./((x.lats-phi0)*(x.lats-phi1)))
     ug = np.zeros((nlats), np.float)
     ug = np.where(np.logical_and(x.lats < phi1, x.lats > phi0), u1, ug)
     ug.shape = (nlats, 1)
     ug = ug*np.ones((nlats, nlons), dtype=np.float)     # broadcast to shape (nlats, nlonss)
     # height perturbation.
     hbump = hamp*np.cos(lats)*np.exp(-((lons-np.pi)/alpha)**2)*np.exp(-(phi2-lats)**2/beta)
  
     # initial vorticity, divergence in spectral space
     vrtspec, divspec = x.getvrtdivspec(ug, vg)
     print(vrtspec.shape,divspec.shape)
     # create hyperdiffusion factor
     hyperdiff_fact = np.exp((-dt/efold)*(x.lap/x.lap[-1])**(ndiss/2))
  
     # solve nonlinear balance eqn to get initial zonal geopotential,
     # add localized bump (not balanced).
     vrtg = x.spectogrd(vrtspec)

     tmpg1 = ug*(vrtg+f)
     tmpg2 = vg*(vrtg+f)
     print(tmpg1.shape)
     tmpspec1, tmpspec2 = x.getvrtdivspec(tmpg1, tmpg2)
     print(tmpspec1.shape)
     tmpspec2 = x.grdtospec(0.5*(ug**2+vg**2))
     phispec = x.invlap*tmpspec1 - tmpspec2
     phig = grav*(hbar + hbump) + x.spectogrd(phispec)
     phispec = x.grdtospec(phig)


     # initialize spectral tendency arrays
     ddivdtspec = np.zeros(vrtspec.shape+(3,), np.complex)
     dvrtdtspec = np.zeros(vrtspec.shape+(3,), np.complex)
     dphidtspec = np.zeros(vrtspec.shape+(3,), np.complex)
     nnew = 0
     nnow = 1
     nold = 2
  
     # time loop.
     time1 = time.time()
     for ncycle in range(itmax+1):
         t = ncycle*dt
         # get vort, u, v, phi on grid
         vrtg = x.spectogrd(vrtspec)
         ug, vg = x.getuv(vrtspec, divspec)
         phig = x.spectogrd(phispec)
         
         print("t=%6.2f hours: min/max %6.2f, %6.2f" % (t/3600., vg.min(), vg.max()))
  
         # compute tendencies.
         tmpg1 = ug*(vrtg+f)
         tmpg2 = vg*(vrtg+f)
         ddivdtspec[:, nnew], dvrtdtspec[:, nnew] = x.getvrtdivspec(tmpg1, tmpg2)
         dvrtdtspec[:, nnew] *= -1
  
         tmpg = x.spectogrd(ddivdtspec[:, nnew])
         tmpg1 = ug*phig
         tmpg2 = vg*phig
         tmpspec, dphidtspec[:, nnew] = x.getvrtdivspec(tmpg1, tmpg2)
         dphidtspec[:, nnew] *= -1
  
         tmpspec = x.grdtospec(phig+0.5*(ug**2+vg**2))
         ddivdtspec[:, nnew] += -x.lap*tmpspec
  
         # update vort, div, phiv with third-order adams-bashforth.
         # forward euler, then 2nd-order adams-bashforth time steps to start.
         if ncycle == 0:
             dvrtdtspec[:, nnow] = dvrtdtspec[:, nnew]
             dvrtdtspec[:, nold] = dvrtdtspec[:, nnew]
             ddivdtspec[:, nnow] = ddivdtspec[:, nnew]
             ddivdtspec[:, nold] = ddivdtspec[:, nnew]
             dphidtspec[:, nnow] = dphidtspec[:, nnew]
             dphidtspec[:, nold] = dphidtspec[:, nnew]
         elif ncycle == 1:
             dvrtdtspec[:, nold] = dvrtdtspec[:, nnew]
             ddivdtspec[:, nold] = ddivdtspec[:, nnew]
             dphidtspec[:, nold] = dphidtspec[:, nnew]
  
         vrtspec += dt*(
             (23./12.)*dvrtdtspec[:, nnew] - (16./12.)*dvrtdtspec[:, nnow]
             + (5./12.)*dvrtdtspec[:, nold])
         divspec += dt*(
             (23./12.)*ddivdtspec[:, nnew] - (16./12.)*ddivdtspec[:, nnow]
             + (5./12.)*ddivdtspec[:, nold])
         phispec += dt*(
             (23./12.)*dphidtspec[:, nnew] - (16./12.)*dphidtspec[:, nnow]
             + (5./12.)*dphidtspec[:, nold])
         # implicit hyperdiffusion for vort and div.
         vrtspec *= hyperdiff_fact
         divspec *= hyperdiff_fact
         # switch indices, do next time step.
         nsav1 = nnew
         nsav2 = nnow
         nnew = nold
         nnow = nsav1
         nold = nsav2
  
     time2 = time.time()
     print("CPU time = ", time2-time1)
  
     # make a contour plot of potential vorticity in the Northern Hem.
     fig = plt.figure(figsize=(12, 4))
     # dimensionless PV
     pvg = (0.5*hbar*grav/omega)*(vrtg+f)/phig
     print("max/min PV", pvg.min(), pvg.max())
     lons1d = (180./np.pi)*x.lons-180.
     lats1d = (180./np.pi)*x.lats
     levs = np.arange(-0.2, 1.801, 0.1)
  
     cs = plt.contourf(lons1d, lats1d, pvg, levs, extend="both", cmap="nipy_spectral")
     cb = plt.colorbar(cs, orientation="horizontal")
     cb.set_label("potential vorticity")
  
     plt.grid()
     plt.xlabel("degrees longitude")
     plt.ylabel("degrees latitude")
     plt.xticks(np.arange(-180, 181, 60))
     plt.yticks(np.arange(-5, 81, 20))
     plt.axis("equal")
     plt.axis("tight")
     plt.ylim(0, lats1d[0])
     plt.title("PV (T%s with hyperdiffusion, hour %6.2f)" % (ntrunc, t/3600.))
     plt.savefig("output_swe.pdf")
     plt.show()

main()
