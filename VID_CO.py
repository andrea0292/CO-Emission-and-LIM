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

z_ary = np.loadtxt('/Users/andreacaputo/Desktop/Phd/lim-master 4/zary_forbiasav.dat')
bias_average_ary = np.loadtxt('/Users/andreacaputo/Desktop/Phd/lim-master 4/biasav_ary.dat')

# let's interpolate
toint= interp1d(z_ary,bias_average_ary)

def biasav(z):
    return 1.*toint(z)

# create instance of the class "Class"
LambdaCDM = Class()
# pass input parameters
m1=0.06/3
m2=0 #0.06/3
m3= 0 #0.06/3

LambdaCDM.set({'N_ncdm':3})
LambdaCDM.set({'m_ncdm':str(m1)+','+str(m2)+','+str(m3)})
LambdaCDM.set({'omega_b':0.022032,'omega_cdm':0.12038,'h':0.67556,'A_s':2.215e-9,'n_s':0.9619,'tau_reio':0.0925})
LambdaCDM.set({'output':'mPk','P_k_max_1/Mpc':100, 'z_max_pk':10.})
# run class
LambdaCDM.compute()

def rhocrit(z):
    return cosmo.critical_density(z).to(u.Msun*u.Mpc**-3) #Msun/Mpc^3

def R(M,z): # Pass the mass in Msun
    
    return (3.0* M/(4.0*np.pi* rhocrit(z).value))**(1.0/3.0)

def W_tophat(k,R):
    
    return 3*(np.sin(k*R) - k*R*np.cos(k*R))/(k*R)**3

def Pk_with_tophat(R,k,z):
    
    return LambdaCDM.pk(k,z) * W_tophat(k, R)**2 #*cosmo.h**3

def sigmaM(M,z):# Mass passed in solar masses
    
    kk = np.logspace(-4,np.log10(100),1000) # k in 1/Mpc
    
    scaleR = R(M,z)
    
    integrand = np.array([Pk_with_tophat(scaleR,ki,z)*ki**2/(2.*np.pi**2) for ki in kk])
    
    sigma = np.sqrt(np.trapz(integrand,kk))
    
    return sigma

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


class VIDco:
    def __init__(self):
        """
        Container class to compute the VID of CO intensity map following arXiv 1609.01728

        """

        # Set constants
        self.phistar = 2.8e-10 * u.Lsun**-1*u.Mpc**-3
        self.alfa = -1.87
        self.Lstar = 2.1e6 * u.Lsun  # in Watt
        self.Lmin = 5e3 * u.Lsun # follow arXiv 1609.01728

    def XLT(self, nu,z):
    
        return (const.c**3 * (1+z)**2 / (8 * np.pi * const.k_B * nu**3 * cosmo.H(z))).to(u.K *u.Mpc**3 / u.W)

    def phi(self, L): # luminosity function

        return self.phistar * (L / self.Lstar)**self.alfa * np.exp(-L /self.Lstar - self.Lmin/L)

    def phitoInt(self, L): # Extract the values without units to integrate
    
        return self.phi(L * u.L_sun).value

    
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


    def Nch(self, Delta_nu, dnu):
    
        #Number of frequency channels, rounded if dnu does not divide evenly into Delta_nu
        
        return np.round((Delta_nu/dnu).decompose())

    def beam_width(self, beam_FWHM):
        
        #Beam width defined as 1-sigma width of Gaussian beam profile
        
        return beam_FWHM*0.4247

    
    def Nside(self, Omega_field, beam_FWHM):
        
        #Number of pixels on a side of a map.  Pixel size is assumed to be one beam FWHM on a side. 
        # Rounded if FWHM does not divide evenly into sqrt(Omega_field)
    
        theta_side = np.sqrt(Omega_field)
        return np.round((theta_side/self.beam_width(beam_FWHM)).decompose())
    

    def Npix(self, Omega_field, beam_FWHM):
        
        # Number of pixels in a map
        
        return self.Nside(Omega_field, beam_FWHM)**2
    
    def Nvox(self, Omega_field, beam_FWHM, Delta_nu, dnu):
       
        #Number of voxels in a map
        
        return self.Npix(Omega_field, beam_FWHM)*self.Nch(Delta_nu, dnu)[0]

    def Sfield(self, Omega_field, z):
        
        #Area of single field in the sky in Mpc**2
        
        return (r0(z)**2*(Omega_field/(1.*u.rad**2))).to(u.Mpc**2)
        
    def Lfield(self, nu, nuObs, Delta_nu ):
        
        #Depth of a single field

        z_min = (nu/(nuObs+Delta_nu/2.)-1).value
        z_max = (nu/(nuObs-Delta_nu/2.)-1).value
        dr_los = (r0(z_max)-r0(z_min))
    
        return dr_los #*u.Mpc


    def Vfield(self, Omega_field, nu, nuObs, Delta_nu):
        
        #Comoving volume of a single field
    
        z = (nu - nuObs) / nuObs
        
        return self.Sfield(Omega_field, z) * self.Lfield(nu, nuObs, Delta_nu )


    def Vvox(self, Omega_field, nu, nuObs, Delta_nu, dnu, beam_FWHM): 
    
        # Comoving volume of a single voxel
    
        return self.Vfield(Omega_field, nu, nuObs, Delta_nu) / self.Nvox(Omega_field, beam_FWHM, Delta_nu, dnu)


    def P1(self, T, Omega_field, nu, nuObs, Delta_nu, dnu, beam_FWHM): # pass the Vvox in Mpc**3; this is the probability of observing a

        # temperature T in a voxel with exactly one source

        nbar0 = quad(self.phitoInt, 10, 1e8 )[0] * u.Mpc**-3 # the integral of phi(L); it will be in Mpc^-3
    
        z = (nu - nuObs) / nuObs
        VVox = self.Vvox(Omega_field, nu, nuObs, Delta_nu, dnu, beam_FWHM)
    
        LT = ((T * u.K) * VVox  /  self.XLT(nu, z)).to(u.Lsun) 

        p = VVox / nbar0 * self.phi(LT) /  (self.XLT(nu, z).to(u.Mpc**3 * u.K / u.Lsun))
    
        return p.value

    def nufunction(self, M, z):
    
        deltac = 1.69
    
        return deltac / sigmaM(M,z)

    def bias(self, M, z):
    
        deltac = 1.69
    
        return 1 + (self.nufunction(M, z) -1 )/ deltac

    def sigma_G(self, which,dnu, nuObs, beam_FWHM, z, nmu): # rms of fluctuations in a voxel 
    
    # The frequency need to be passed with units, e.g 30 * u.GHz;
    # The beam_FWHM needs units too, for example 3 * u.arcmin.
    
    # nmu is the number of bins in the mu variable, which is the cosine of the angle between 
    #the mode k and the LOS
    
        mu = np.linspace(-1,1,1000+1) # array of mu (cos theta)
    
    
        sigmapar = sigma_par(dnu, nuObs, z).value
        sigmaperp = sigma_perp(beam_FWHM, z).value
    
        #kvalues, and kpar, kperp. Power spectrum in observed redshift
        k = np.logspace(-2,2,128)
        ki,mui = np.meshgrid(k,mu)
    
    
        #Pk = np.array([LambdaCDM.pk(kk,z)*cosmo.h**3 for kk in k]) # in Mpc^-3
    
        if which == 'DM':
            def Power(z,k):
                return LambdaCDM.pk(k,z)  #*cosmo.h**3
        
        if which == 'LIM':
            def Power(z,k):
                return LambdaCDM.pk(k,z) * biasav(z)**2#*cosmo.h**3 
         
        Pk = np.array([Power(z,kk) for kk in k]) # in Mpc^-3
    
        kpar = ki*mui
        kperp = ki*np.sqrt(1.-mui**2.)
            
        #Gaussian window for voxel -> FT
        Wkpar2 = np.exp(-((kpar* sigmapar)**2))
        Wkperp2 = np.exp(-((kperp* sigmaperp)**2))
        Wk2 = Wkpar2*Wkperp2
            
        #Compute sigma_G
        integrnd = Pk*Wk2*ki**2/(4.*np.pi**2)
        integrnd_mu = np.trapz(integrnd,mu,axis=0)
        sigma = np.sqrt(np.trapz(integrnd_mu,ki[0,:]))
            
        return sigma

    def lognormal_Pmu(self, mu,Nbar,sig_G):
        '''
        Lognormal probability distribution of mean galaxy counts mu, with width
        set by sig_G.  This function gives the PDF of the underlying lognormal
        density field, can be combined with a Poisson distribution to get a model
        for P(Ngal) as explained in Breyes et al. 2017
        '''
        Pln = (np.exp(-(np.log(mu/Nbar)+sig_G**2/2.)**2/(2*sig_G**2)) / (mu*np.sqrt(2.*np.pi*sig_G**2.)))

        return Pln

    def PofN(self, sigmaG, Vvox, N):
    
       
        #Probability of a voxel containing N galaxies.  We follow Breysse et al. 2017 with a lognormal + Poisson model

        nbar0 = quad(self.phitoInt, 10, 1e8 )[0] * u.Mpc**-3 # the integral of phi(L); it will be in Mpc^-3

        nbar = nbar0.value
        
        Nbar = nbar * Vvox
    
        logMuMin = np.log10(nbar * Vvox )-20*sigmaG
        logMuMax = np.log10(nbar * Vvox )+5*sigmaG
    
        mu = np.logspace(logMuMin,logMuMax,10**4)
        mu2,N = np.meshgrid(mu, N) # Keep arrays for fast integrals
    
        Pln = self.lognormal_Pmu(mu2, Nbar, sigmaG)

        P_poiss = poisson.pmf(N,mu2)
                
        return np.trapz(P_poiss*Pln,mu)[0]

    def Pnoise(self, Temperature, sigmaN):
    
        # SigmaN is the survey sensitivity, pass it in micro-Kelvin directly.
    
        return 2/np.sqrt(2*np.pi*sigmaN**2) *np.exp(-Temperature**2/2/(sigmaN*1e-6)**2)

    def PnLIM(self, T, N, nmu, Omega_field, nu, nuObs, Delta_nu, dnu, beam_FWHM):
    
        dT = 1e-3
        z = (nu - nuObs) / nuObs
    
        Tarray = np.linspace(T,T + dT, 100)
    
        siG = self.sigma_G('LIM',dnu, nuObs, beam_FWHM, z, nmu)
        VVox = self.Vvox(Omega_field, nu, nuObs, Delta_nu, dnu, beam_FWHM).value
    
        P1_ary_aux = np.array([1e-6 * self.P1(T * 1e-6, Omega_field, nu, nuObs, Delta_nu, dnu, beam_FWHM) for T in Tarray])
        fP1 = np.fft.fft(P1_ary_aux)*(dT)
    
        sumFourier = np.zeros(len(Tarray))
    
        for ii in range(1,N+1):
            sumFourier = sumFourier + fP1**(ii) *self.PofN(siG, VVox, ii)
        
        transformed = ((np.fft.ifft(sumFourier)/(dT)).real)
        
        return np.trapz(transformed,  Tarray)/dT

    def NoiseLIM(self, Temperature, sigmaN): # We built the noise function and noise array, to then convolve with the signal
    
        # SigmaN is the survey sensitivity, pass it in micro-Kelvin directly.
    
        return 2/np.sqrt(2*np.pi*sigmaN**2) *np.exp(-Temperature**2/2/(sigmaN)**2)

    def NoiseAry(self, Tarray, sigmaN):
    
        noise_ary = np.array([self.NoiseLIM(Te, sigmaN) for Te in Tarray])
    
        return noise_ary

    def Add_things(self, P1,P2,dT): # Convolution of two arrays you pass to the function

        fP1 = np.fft.fft(P1) * dT
        fP2 = np.fft.fft(P2) * dT
        
        jointfP = fP1*fP2
        
        return ((np.fft.ifft(jointfP)/dT).real) 
    