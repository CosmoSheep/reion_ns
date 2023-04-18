import numpy as np
import lya_convertor as lya_c
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.integrate import trapz

"""
    This class is in charge of describing SKA1-LOW for grabing info needed for noise. The technical specs come from Francisco's 2014 paper and other references. This one is the second version, which accounts for the lack of correlations between the antennas in each station.
    
    Note that this version has very low sensitivity, in fact it has a pretty subpar number density of baselines!
    
"""

class skalow(object):
    
    def __init__(self, t_int, b, Omega_m, h, Omega_r):
        # let's grab the bandwidth
        self.band_MHz = b # in MHz
        # grabbing total integration time in h and transform to s
        self.t_int = t_int * 3600.
        # physical dish area
        self.D_phys = 40.0 # meters
        # square root of effective dish area
#        self.D_eff = np.sqrt(0.7) * self.D_phys # in meters
        # total survey area (100 sq. degrees coming from the proposed survey for low-z in SKA
        # Red book)
        self.S_area = 100 * (np.pi / 180.)**2 # sq. radians
        # number of receivers = 224 stations * 256 dipole antennas in each
        self.N_s = 224 * 2 # stations in the core
        self.Total_N_s = 512 * 2 # total stations
        self.N_a = 256 # this is the number of antennas per station, not being used for anything
        # the data that we grabbed manually
        self.U_over_nu, self.n_U_x_nu_nu = np.loadtxt('density_baseline.csv', unpack=True)
        # for system noise we need the comoving distance
        self.lya_c = lya_c.lya_convert(Omega_m, h, Omega_r)
        # speed of ligth in km/s
        self.c_kms = 2.99979e5
        self.D_max = 1000 # core has radius of 500 meters
        # for bug catching
        self.verbose = False

    def D_eff(self):
        """
            Returns the effective dish diameter
        """
        return np.sqrt(0.7) * self.D_phys

    def T_sky(self, nu_obs):
        """
            Returns the sky temperature in Kelvins. The input should be in Megahertz.
        """
        return 60 * pow(300. / nu_obs, 2.55)
        
        
    def T_sys(self, nu_obs):
        """
            Returns the system temperature for SKA1-LOW. The input should be in MHz and the output in mili Kelvins
        """
        T_tel = 40 # in Kelvins
        # receiver temp in Kelvins
        T_rcvr = 0.1 * self.T_sky(nu_obs) + T_tel
        T_sys = self.T_sky(nu_obs) + T_rcvr
        return T_sys * 1000.
        
    def FOV(self, lambda_obs):
        """
            Returns the effective field of view
        """
        return pow((lambda_obs / self.D_eff()), 2.)
        
        
    def square_uni_density_sk(self, u, lambda_obs):
        """
            Returns a square close package number density of baselines in the uv plane. We use the usual approx.
            
            Function is just for reference, it is not being used in the main code.
        """
        # using the core, where power spectrum is better measured
        D_max = 1000. # in meters
        u_max = D_max / lambda_obs
        if u <= u_max:
            return self.N_s**2 / (2. * np.pi * pow(u_max, 2.))
        else:
            return 0.0000001
        
    def number_density(self, u, lambda_obs):
        """
            Returns number density of baselines in the uv plane
        """
        c_mMHz = 2.99979e2 # speed of light in meter MegaHertz
        nu_obs = c_mMHz / lambda_obs
        u_array = self.U_over_nu * nu_obs
        n_u_array = self.n_U_x_nu_nu / nu_obs / nu_obs
        n_u_interp = interp1d(u_array, n_u_array, fill_value='extrapolate')
        return n_u_interp(u)
        
    def normalize(self, lambda_obs):
        """
           Compute normalization of the data used in Francisco's work
        """
        c_mMHz = 2.99979e2 # speed of light in meter MegaHertz
        nu_obs = c_mMHz / lambda_obs
        u_array = self.U_over_nu * nu_obs
        n_u_array = self.n_U_x_nu_nu / nu_obs / nu_obs
#        print('U: ',u_array)
#        print('n: ', n_u_array)
        norm = trapz(2. * np.pi * u_array * n_u_array, u_array)
        proper_SKA = 0.5 * self.N_s * (self.N_s - 1)
        return proper_SKA / norm

 
    
    
    def plot_density_baselines(self, z):
        """
            Quick check that the online extractor actually works, data obtained from Villaescusa-Navarro et al. (2015)
            
            Also, for debugging looks at the sq. uni. approx.
        """
        lambda_obs = 0.21 * (1. + z)
        nu_obs = 1420. / (1. + z)
        sq_uni = [] # This is N_s^2
        sq_uni_d = [] # N_s^4
        for i in self.U_over_nu:
            sq_uni_d.append(self.square_uni_density_sk(i, lambda_obs) * nu_obs**2)
        
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_subplot(111)
        ax1.plot(self.U_over_nu, self.n_U_x_nu_nu, label='FVN 2014')
        ax1.plot(self.U_over_nu, sq_uni_d, label=r'Sq. Uni. 224 stations')
        ax1.set_xlabel(r'$U/\nu$    [MHz$^{-1}$]', fontsize=14)
        ax1.set_ylabel(r'$n(U) \times \nu^2$    [MHz$^2$]', fontsize=14)
        ax1.set_title('Baseline density distribution for SKA1-LOW', fontsize=14)
#        ax1.set_xlim(1.e-2, 1.e3)
#        ax1.set_ylim(1.e-6,1.e5)
        ax1.legend(loc='best')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        plt.show()
        
        
        
        u_array = self.U_over_nu * nu_obs
        n_u_array = self.n_U_x_nu_nu / nu_obs**2
        # deal with normalization
        norm_fact = self.normalize(lambda_obs)
        sq_array = []
        sq_array_d = []
        nu_den = []
        fixed_nu_den = []
        new_sq = []
        pusk = []
        for u in u_array:
            sq_array_d.append(self.square_uni_density_sk(u, lambda_obs))
            nu_den.append(self.number_density(u, lambda_obs) * norm_fact)
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_subplot(111)
#        print(len(u_array), len(n_u_array))
#        ax1.plot(u_array, n_u_array)
        ax1.plot(u_array, nu_den, label=r'FNV 2014 (224 stations)')
        ax1.plot(u_array, sq_array_d, label=r'Sq. Uni. ~ 224 stations')
        ax1.set_xlabel(r'$U$', fontsize=14)
        ax1.set_ylabel(r'$n(U)$', fontsize=14)
        ax1.set_title('Baseline density distribution for SKA1-LOW', fontsize=14)
    #        ax1.set_xlim(1.e-2, 1.e3)
#        ax1.set_ylim(1.e-5,1.e4)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.legend(loc='best')
        plt.show()
        return 1
    
    def eff_area_SKA_LOW(self, nu_MHz):
        """
          Returns the effective collective area in sq. meters for the SKA1-LOW survey, connecting Francisco's to https://www.cambridge.org/core/services/aop-cambridge-core/content/view/161A30708237B560B3869F336A4DCB3F/S1323358019000511a.pdf/cosmology-with-phase-1-of-the-square-kilometre-array-red-book-2018-technical-specifications-and-performance-forecasts.pdf
        """
        # critical collective area
        A_e_crit = 3.2 * 256 # sq. m area per antenna times number of antennas in a station
        # critical frequency in Mega Hertz
        nu_crit_MHz = 110.
        if (nu_MHz > nu_crit_MHz):
            return A_e_crit * pow(nu_crit_MHz / nu_MHz,2)
        elif (nu_MHz <= nu_crit_MHz):
            return A_e_crit

    def noise_power_Mpc(self, z, k_Mpc, mu):
        """
            Returns the noise power spectrum in mK^2 Mpc^3.
            
            The inputs are redshift, wavenumber in 1/Mpc and the angle between the line of sigth mu
        """
        lambda_obs = 0.21 * (1. + z) # note that this is in meters
        nu_obs = 1420 / (1. + z) # in MHz, I got lazy so defined both hehe
        # comoving distance in Mpc
        D_c_Mpc = self.lya_c.comoving_dist(z)
        # hubble parameter in km/s/Mpc
        H_kms_Mpc = self.lya_c.hubble(z)
        # angular resolution factor
        ang_factor = lambda_obs**2 / self.eff_area_SKA_LOW(nu_obs) # [sq. m] / [sq. m] = dimensionless
        # FOV factor
        f_factor = self.S_area / self.FOV(lambda_obs) # dimensionless
        # number density stuff
        # first get the right k perp
        kt_Mpc = k_Mpc * np.sqrt(1. - mu**2)
        u = kt_Mpc * D_c_Mpc / (2. * np.pi)
        # grab density
        n = self.number_density(u, lambda_obs)
        # and normalize it to the number of baselines that is in the code. 
        norm_factor = self.normalize(lambda_obs)
        # then density becomes
        n *= norm_factor
        den_factor = 2. * self.t_int * n # [s]
        # construct noise, note that I transform a lambda obs to km due to hubble's units
        P_N = (self.T_sys(nu_obs))**2 * D_c_Mpc**2 * (lambda_obs / 1000.) * (1. + z) / H_kms_Mpc * ang_factor**2 / den_factor * f_factor
        if self.verbose == 1:
            print('Lambda obs in km ', lambda_obs / 1000.)
            print('Nu obs in MHz ', nu_obs)
            print('Comoving distance in Mpc ', D_c_Mpc)
            print('Hubble in km/s/Mpc ', H_kms_Mpc)
            print('FOV is, in sq. deg, ', self.FOV(lambda_obs) * pow(180. / np.pi, 2))
            print('kt in 1/Mpc is ', kt_Mpc)
            print('The u ', u)
            print('Integration time ', self.t_int)
            print('Number density for that kt ', n)
            print('T system in mili Kelvins ', self.T_sys(nu_obs))
#            prefrator = (self.T_sys(nu_obs))**2 * D_c_Mpc**2 * (lambda_obs / 1000.) * (1. + z) / H_kms_Mpc
            prefrator = D_c_Mpc**2 * (lambda_obs / 1000.) * (1. + z) / H_kms_Mpc
            print('Prefactor ', prefrator)
            print('1 / den_factor ', 1. / den_factor)
            print('S_area / FOV ', f_factor)
            print('(Lambda_obs^2 / A_e)^2 ', ang_factor**2)
            print('P_N ', prefrator * f_factor * ang_factor**2 / den_factor)
            print('Delta^2_N ', P_N * pow(k_Mpc, 3.) / (2.0 * np.pi))
        return P_N
        


    def noise_power_square_uni_Mpc(self, z, k_Mpc, mu):
        """
            Returns the noise power spectrum in mK^2 Mpc^3.
            
            The inputs are redshift, wavenumber in 1/Mpc and the angle between the line of sigth mu.
            
            Function is just for reference, it is not being used in the main code.
        """
        lambda_obs = 0.21 * (1. + z) # note that this is in meters
        nu_obs = 1420 / (1. + z) # in MHz, I got lazy so defined both hehe
        # comoving distance in Mpc
        D_c_Mpc = self.lya_c.comoving_dist(z)
        # hubble parameter in km/s/Mpc
        H_kms_Mpc = self.lya_c.hubble(z)
        # angular resolution factor
        ang_factor = lambda_obs**2 / self.eff_area_SKA_LOW(nu_obs) # [sq. m] / [sq. m] = dimensionless
        # FOV factor
        f_factor = self.S_area / self.FOV(lambda_obs) # dimensionless
        # number density stuff
        # first get the right k perp
        kt_Mpc = k_Mpc * np.sqrt(1. - mu**2)
        u = kt_Mpc * D_c_Mpc / (2. * np.pi)
        n = self.square_uni_density_sk(u, lambda_obs)
        den_factor = 2. * self.t_int * n # [s]
        # construct noise, note that I transform a lambda obs to km due to hubble's units
        P_N = (self.T_sys(nu_obs))**2 * D_c_Mpc**2 * (lambda_obs / 1000.) * (1. + z) / H_kms_Mpc * ang_factor**2 / den_factor * f_factor
        if self.verbose == 1:
            print('Lambda obs in km ', lambda_obs / 1000.)
            print('Nu obs in MHz ', nu_obs)
            print('Comoving distance in Mpc ', D_c_Mpc)
            print('Hubble in km/s/Mpc ', H_kms_Mpc)
            print('FOV is, in sq. deg, ', self.FOV(lambda_obs) * pow(180. / np.pi, 2))
            print('kt in 1/Mpc is ', kt_Mpc)
            print('The u ', u)
            print('Integration time ', self.t_int)
            print('Number density for that kt ', n)
            print('T system in mili Kelvins ', self.T_sys(nu_obs))
        #            prefrator = (self.T_sys(nu_obs))**2 * D_c_Mpc**2 * (lambda_obs / 1000.) * (1. + z) / H_kms_Mpc
            prefrator = D_c_Mpc**2 * (lambda_obs / 1000.) * (1. + z) / H_kms_Mpc
            print('Prefactor ', prefrator)
            print('1 / den_factor ', 1. / den_factor)
            print('S_area / FOV ', f_factor)
            print('(Lambda_obs^2 / A_e)^2 ', ang_factor**2)
            print('P_N ', prefrator * f_factor * ang_factor**2 / den_factor)
            print('Delta^2_N ', P_N * pow(k_Mpc, 3.) / (2.0 * np.pi))
        return P_N

    

    def survey_vol_Mpc(self, z):
        """
            Volume of the survey, for auto it is the 21cm, for cross it would be both surveys have the same volume.
            
            Units of Mpc^3
        """
        # frequency
        nu_MHz = 1420.
        # comoving distance
        Dc_Mpc = self.lya_c.comoving_dist(z)
        # comoving distance associated with bandwith of the instrument
        Delta_D_Mpc = self.c_kms / (self.lya_c.h * 100.) / np.sqrt(self.lya_c.Omega_m) / nu_MHz * np.sqrt(1. + z) * self.band_MHz
        # note bandwidth needs to be in MHz just like the 21cm frequency
        nu_obs = nu_MHz / (1. + z)
        lambda_obs = 0.21 * (1. + z)
        """ Sorry Heyang I will clean up all the lambda_emit and nu_emit later haha"""
        # for area of a single dish we use the eff area
        # ratio to account for the angular resolution of the telescope in Fourier space
        ratio_single = lambda_obs * lambda_obs / self.eff_area_SKA_LOW(nu_obs)
        # now that would be the expression for a single field interferometer survey, but we have many fields
        ratio_all = self.S_area / self.FOV(lambda_obs)
        return Dc_Mpc**2 * Delta_D_Mpc * ratio * ratio_all


    def checking_lightcone(self, z_1, band_MHz):
        """
            Checking that our redshift bins are small enough to avoid lightcone effects, according to Greig & Mesinger we need bins with width less than 20 MHz (usually people use 8 MHz though)
        """
        nu_MHz = 1420.
        a = band_MHz / nu_MHz
        # given z1 begin the start of the bin then the end would be
        z_2 = (1 + z_1) / ((1 + z_1) * a + 1) - 1
        print('printing end redshift and redshift bin size')
        print('Remember that we want redshift bin size to be larger than our choice ~ 0.1.')
        return z_2, z_1 - z_2
