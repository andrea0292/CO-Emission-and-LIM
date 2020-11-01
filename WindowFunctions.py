import numpy as np
import astropy.units as u
from astropy.cosmology import Planck13 as cosmo
from astropy import constants as const
from scipy.special import legendre
import camb
import pickle
from scipy.integrate import quad
from scipy.integrate import fixed_quad
from scipy.integrate import romberg
from scipy.fft import fft,ifft
import matplotlib.gridspec as gridspec
from scipy.stats import poisson
from classy import Class

from scipy.interpolate import interp1d
from scipy.interpolate import interp2d


def r0(z):
    
    return cosmo.comoving_distance(z) # in Mpc

def beam_width(beam_FWHM):
        
    #Beam width defined as 1-sigma width of Gaussian beam profile
        
    return beam_FWHM*0.4247

# Here we have the characteristic resolution limits in the radial and transverse directions,
# as can be seen in Eq.12 of --> https://arxiv.org/pdf/1907.10067.pdf


def sigma_perp(beam_FWHM, z):
    
    # transverse mode resolution
    
    return (r0(z)*(beam_width(beam_FWHM)/(1*u.rad))).to(u.Mpc) # in Mpc


def sigma_par(dnu, nuObs, z): 
        
    # cutoff for line-of-sight modes
        
    return (const.c*dnu*(1+z)/(cosmo.H(z)*nuObs)).to(u.Mpc) # in Mpc


class WindowFunctions:
    def __init__(self):
        """
        Container class to compute window functions due to resolution and volume effects. See for example https://arxiv.org/pdf/1907.10067.pdf

        """

        # Set constants
        self.fromHztoeV = 6.58e-16
        self.gramstoeV = 1 / ( 1.78 * 1e-33)
        self.mtoev = 1/(1.97 * 1e-7) 
        self.H0 = cosmo.H(0).value * 1e3 / (1e3 * const.kpc.value) #expressed in 1/s
        self.rhocritical = cosmo.critical_density(0).value * self.gramstoeV /(1e-2)**3 # eV/m**3
        self.Om0 = cosmo.Om0 #total matter 
        self.OLambda0 = cosmo.Ode0  # cosmological constant
        self.DM0 = self.Om0 - cosmo.Ob0 # dark matter
        self.evtonJoule = 1.60218 * 1e-10 # from eV to nJ
        self.evtoJoule = 1.60218 * 1e-19 # from eV to J
        self.h  = 0.6766
        self.Mpc = 1e3 * const.kpc.value
        self.zmin = 0.001
        self.zmax = 30.001
        self.zbins = 301
        self.h = cosmo.h
    
    def Sfield(self, Omega_field, z):
        
        #Area of single field in the sky in Mpc**2
        
        return (r0(z)**2*(Omega_field/(1.*u.rad**2))).to(u.Mpc**2)

    def Lfield(self, nu, nuObs, Delta_nu ):
        
        #Depth of a single field

        z_min = (nu/(nuObs+Delta_nu/2.)-1).value
        z_max = (nu/(nuObs-Delta_nu/2.)-1).value
        dr_los = (r0(z_max)-r0(z_min))
    
        return dr_los #*u.Mpc


    def kparmin(self, nu, Delta_nu, nuObs):
    
        return 2*np.pi / (self.Lfield(nu, nuObs, Delta_nu ))

    def kpermin(self, Omega_field, z):
    
    
        return 2 * np.pi / np.sqrt(self.Sfield(Omega_field, z))


    def Wres(self, k, mu, dnu, nuObs,beam_FWHM, z): # Window function for resolution effects
    
        exponent = (-k**2 * (sigma_perp(beam_FWHM, z)**2 * (1-mu**2) + sigma_par(dnu, nuObs, z)**2 * mu**2))
    
        return np.exp(exponent)

    def Wvol(self, k, mu,nu, Delta_nu, nuObs, Omega_field): # Window function for volume effects 
    
        z = (nu - nuObs) / nuObs
    
        Nper = 2
        Npar = 2 # post reionization scenarios
    
        return (1 - np.exp(-(k/ Nper / self.kpermin(Omega_field, z))**2 *(1-mu**2))) * ( 1 - np.exp(-(k / self.kparmin(nu, Delta_nu, nuObs)/Npar)**2 * mu**2))

    def Window_tot(self, k, mu,nu, Delta_nu, nuObs,dnu, Omega_field,beam_FWHM, z):
    
        return self.Wvol(k, mu,nu, Delta_nu, nuObs, Omega_field)*self.Wres(k, mu, dnu, nuObs,beam_FWHM, z) 

    def WindowMission(self, k, mu, MISSION): # define a window function to which you directly pass a dictionary

        nu = MISSION["nu"]
        nuObs = MISSION["nuObs"] 
        dnu = MISSION["dnu"]
        Delta_nu = MISSION["deltanu"]
        Omega_field = MISSION["omega"]
        beam_FWHM = MISSION["beam"]


        zmission = (nu - nuObs).value / nuObs.value

        return self.Window_tot(k, mu,nu, Delta_nu, nuObs,dnu, Omega_field,beam_FWHM, zmission)

    
    
