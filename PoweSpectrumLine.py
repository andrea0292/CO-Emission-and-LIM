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


from colossus.cosmology import cosmology
cosmology.setCosmology('planck15') # Set cosmology for Colossus 
from colossus.lss import bias
from colossus.lss import mass_function

import nbodykit
from nbodykit.lab import *

import WindowFunctions
from WindowFunctions import WindowFunctions
# create an instance of the class for computing Window Functions

Window = WindowFunctions()

cosmo2 = cosmology.Planck15

# define mass functions using Colossus

def MfunctionColossus(m, z):
    return 1/m * cosmo.h**3 * mass_function.massFunction(m*cosmo.h,z, mdef = '200m', model = 'tinker08', q_out = 'dndlnM')


BehrooziFile = np.loadtxt("/Users/andreacaputo/Desktop/Phd/lim-lim-dev/SFR_tables_sfr_release.dat") # data for the CO model
zary_ = np.loadtxt('/Users/andreacaputo/Desktop/Phd/lim-master 4/zaryforbiasav.dat')
bias_avCO = np.loadtxt('/Users/andreacaputo/Desktop/Phd/lim-master 4/Biasav.dat')
sigv_ary= np.loadtxt('/Users/andreacaputo/Desktop/Phd/lim-master 4/sigmav_aryz.dat')

interpolated_ary = interp1d(zary_, bias_avCO)
interpolated_ary2 = interp1d(zary_, sigv_ary)

def bint(z): #interpolated
    return 1. * interpolated_ary(z)

def sigmav(z): # our fiducial value for the velocity dispersion 
    
    return 1. * interpolated_ary2(z)

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

def XLT(nu,z):
    
    return (const.c**3 * (1+z)**2 / (8 * np.pi * const.k_B * nu**3 * cosmo.H(z))).to(u.K *u.Mpc**3 / u.W)

class COpower:
    def __init__(self):
        """
        Container class to compute the PS from lines. In particular here we consider just CO. Adding other lines would amount to add other emission models

        """

    # Here we define the velocity dispersion


    def growthrate(self, z):
        return 1 * cosmo2.scale_independent_growth_rate(z)

    def Kaiser(self, mu, z):
    
        return 1 + self.growthrate(z)/bint(z) * mu**2

    def Rsd(self, k,mu,z): # pass the mode k in Mpc^-1
    
        return self.Kaiser(mu, z) * (1 + 0.5 * (mu * k * sigmav(z) * u.Mpc)**2)**(-1)


    def LTonyLy(self, Mvec, z):
    
        # Behroozi 2013 SFR(M) data
    
        x = BehrooziFile
        zb = np.unique(x[:,0])-1.
        logMb = np.unique(x[:,1])
        logSFRb = x[:,2].reshape(137,122,order='F')
    
        logSFR_interp = interp2d(logMb,zb,logSFRb,bounds_error=False,fill_value=0.)
    
        # Compute SFR(M) in Msun/yr
        logM = np.log10((Mvec.to(u.Msun)).value)
        if np.array(z).size>1:
            SFR = np.zeros(logM.size)
            for ii in range(0,logM.size):
                SFR[ii] = 10.**logSFR_interp(logM[ii],z[ii])
        else:
            SFR = 10.**logSFR_interp(logM,z)
    
        # Compute IR luminosity in Lsun
        LIR = SFR/(1*1e-10) # 1 dmMF 10^10 times SFR/LIR normalization (See Li et al. Eq 1), dimensionless

        alpha = 1.37
        beta = -1.74
    
        # Compute L'_CO in K km/s pc^2
        Lprime = (10.**-beta * LIR)**(1./alpha)
    
        # Compute LCO
        L = (4.9e-5*u.Lsun)*Lprime

        return L

    def rhoaverage(self, z):
    
        Mary = np.logspace(9, 15,1000)
        integrand_ary = np.array([self.LTonyLy(m * u.Msun, z)[0].value *MfunctionColossus(m, z) for m in Mary])
    
        return np.trapz(integrand_ary, Mary)
    
    def Lumi2(self, z):
    
        Mary = np.logspace(9, 15,1000)
        integrand_ary = np.array([self.LTonyLy(m * u.Msun, z)[0].value**2 *MfunctionColossus(m, z) for m in Mary])
    
        return np.trapz(integrand_ary, Mary) * (u.Lsun**2 / u.Mpc**3)

    # Noise term

    def Noiseterm(self, z, nu):
    
        return (XLT(nu, z)**2 * self.Lumi2(z)).to(u.K**2 * u.Mpc**3 )


    # Cluster term for the line emission PS

    def PScluster(self, nu, z, k,mu, Pk): # the bias should be passed at the corresponding redshift 
    
        return (XLT(nu, z)**2 * (self.rhoaverage(z)* u.L_sun/u.Mpc**3)**2 * bint(z)**2 * self.Rsd(k,mu,z)**2 * Pk *u.Mpc**3).to(u.K**2 * u.Mpc**3)

    # This is the total power spectrum, cluster piece + noise term 

    def PtotalLim(self, nu, z, k,mu, Pk):
    
        return self.PScluster(nu, z, k,mu, Pk) + self.Noiseterm(z, nu) 

    # This is the total power spectrum, cluster piece + noise term but passing simply the mission name as a dictionary

    def PtotalLimMission(self, k, Pk, mu, MISSION):

        zline = (MISSION["nu"] - MISSION["nuObs"] ).value / MISSION["nuObs"].value

        return self.PtotalLim(MISSION["nu"], zline, k,mu, Pk)

    # Adding the Window functions 

    def PtotalLimMissionObserved(self, k, Pk, mu, MISSION):

        return self.PtotalLimMission(k, Pk, mu, MISSION) * Window.WindowMission(k, mu, MISSION)

    def PtotalLimObserved(self, nu, k,mu, Pk, Delta_nu, nuObs,dnu, Omega_field,beam_FWHM):

        zline = (nu - nuObs).value / nuObs.value 
    
        return (self.PScluster(nu, zline, k,mu, Pk) + self.Noiseterm(zline, nu)) * Window.Window_tot(k, mu,nu, Delta_nu, nuObs,dnu, Omega_field,beam_FWHM, zline)

    # Now multipoles components 

    def MultipoleNoW(self, k,mui, Pk, nu, nuObs): # mui is the array of mu we will integrate over
    
        z = (nu/nuObs).value - 1.
    
        toint = np.array([self.PtotalLim(nu,z,k,mu, Pk).value for mu in mui])
    
        L2 = legendre(2)(mui)
        L4 = legendre(4)(mui)
    
        PK0 = 0.5*np.trapz(toint, mui,axis=0)
        PK2 = 2.5*np.trapz(toint*L2,mui,axis=0)
        PK4 = 4.5*np.trapz(toint*L4,mui,axis=0)
    
        return  PK0, PK2, PK4 

    def MultipoleNoWmission(self, k,mui, Pk, MISSION): # mui is the array of mu we will integrate over
    
        toint = np.array([self.PtotalLimMission(k, Pk, mu, MISSION).value for mu in mui])
    
        L2 = legendre(2)(mui)
        L4 = legendre(4)(mui)
    
        PK0 = 0.5*np.trapz(toint, mui,axis=0)
        PK2 = 2.5*np.trapz(toint*L2,mui,axis=0)
        PK4 = 4.5*np.trapz(toint*L4,mui,axis=0)
    
        return  PK0, PK2, PK4

    # Define now the multipoles with Window Functions

    def MultipoleLineObs(self, mui, nu, k, Pk, Delta_nu, nuObs,dnu, Omega_field,beam_FWHM):
    
        toint = np.array([self.PtotalLimObserved(nu, k,mu, Pk, Delta_nu, nuObs,dnu, Omega_field,beam_FWHM).value for mu in mui])
    
        L2 = legendre(2)(mui)
        L4 = legendre(4)(mui)
    
        PK0 = 0.5*np.trapz(toint, mui,axis=0)
        PK2 = 2.5*np.trapz(toint*L2,mui,axis=0)
        PK4 = 4.5*np.trapz(toint*L4,mui,axis=0)
    
        return  PK0, PK2, PK4 

    def MultipoleObsWmission(self, mui,k, Pk, MISSION):

        toint = np.array([self.PtotalLimMissionObserved(k, Pk, mu, MISSION).value for mu in mui])
    
        L2 = legendre(2)(mui)
        L4 = legendre(4)(mui)
    
        PK0 = 0.5*np.trapz(toint, mui,axis=0)
        PK2 = 2.5*np.trapz(toint*L2,mui,axis=0)
        PK4 = 4.5*np.trapz(toint*L4,mui,axis=0)
    
        return  PK0, PK2, PK4

    def CllineFinal(self, window, k, Pk, mui, MISSION):

        if window == 'yes': # with window functions 

            return self.MultipoleObsWmission(mui,k, Pk, MISSION)

        if window == 'no': #  without window functions 

            return self.MultipoleNoWmission(k,mui, Pk, MISSION)
