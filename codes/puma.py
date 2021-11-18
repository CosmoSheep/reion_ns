import numpy as np
import lya_convertor as lya_c
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


"""
    This class is in charge of describing PUMA for grabing info needed for noise. The technical specs come from 1810.09572
    
    Still working on this.
    
"""

class puma(object):
    
    def __init__(self, t_int, b):
        # let's grab the bandwidth
        self.band_MHz = b # in MHz
        # grabbing total integration time in h and transform to s
        self.t_int = t_int * 3600.
        # the amplifier noise temperature is
        self.T_ampl = 50 # in Kelvins
        # ground temperature
        self.T_ground = 50 # in Kelvins
        # coupling to vacuum
        self.eta_c = 0.9 # max is 1
        # coupling to sky
        self.eta_s = 0.9 # max is 1
        # aperture efficiency
        self.eta_apert = 0.7 # max is 1
        # coverage of sky
        self.f_sky = 0.5 # max is 1
        # physical dish area
        self.D_phys = 6.0 # meters
        # square root of effective dish area
        self.D_eff = np.sqrt(0.7) * self.D_phys # in meters
        # total survey area
        self.S_area = 4.0 * np.pi * self.f_sky # sq. radians
        # effective collecting area per antenna feed
        self.A_e = np.pi * (self.D_eff / 2.)**2
        # number of receivers
        self.N_s = 256 # hence compact square array will have 256^2
#        """ playing with this for a second it should go back to 256!!! """
        # let's choose a hex-close packing, thus
        self.a = 0.5698 # sq 0.4847
        self.b = -0.5274 #sq -0.3300
        self.c = 0.8358 # sq 1.3157
        self.d = 1.6635 # sq 1.5974
        self.e = 7.3177 # sq 6.8390
        # for system noise we need the comoving distance
        self.lya_c = lya_c.lya_convert()
        # for bug-catching
        self.verbose = True
        
    def T_sky(self, nu):
        """
            Returns the sky temperature in Kelvins. The input should be in Megahertz.
        """
        return pow((nu / 400.), -2.75) * 25. + 2.7
        
    def T_sys(self, nu):
        """
            Returns the system temperature for PUMA. The input should be in MHz and the output is in mili Kelvins
        """
        ampl = self.T_ampl / self.eta_c / self.eta_s
        ground = (1. - self.eta_s) * self.T_ground / self.eta_s
#        return ampl + ground + self.T_sky(nu)
        return (ampl + ground + self.T_sky(nu)) * 1000.
    
    def FOV(self, lambda_obs):
        """
            Returns the effective field of view
        """
        return pow((lambda_obs / self.D_eff), 2.)
        
    def n_b_phys(self, ell):
        """
            Returns  physical number of baselines as a function of physical distance of antennas
        """
        n_0 = (self.N_s / self.D_phys)**2
        L = self.N_s * self.D_phys
        l_fac = ell / L
        mid_fac = (self.a + self.b * l_fac) / (1. + self.c * pow(l_fac, self.d))
        return n_0 * mid_fac * np.exp(-1. * pow(l_fac, self.e))
        
    def number_density(self, u, lambda_obs):
        """
            Returns number density of baselines in the uv plane
        """
        ell = u * lambda_obs
        return self.n_b_phys(ell) * lambda_obs**2
        
    def square_uni_density(self, u, lambda_obs, N_s):
        """
           Returns a square close package number density of baselines in the uv plane, N_s = 256 * 256
        """
        D_min = 6 # meters
        D_max = 256 * 6 * np.sqrt(2) # meters
        u_min = D_min / lambda_obs
        u_max = D_max / lambda_obs
#        return N_s**2 / (2. * np.pi) / (u_max**2 - u_min**2)
        return N_s**2 / (2. * np.pi * u_max**2)
#        return N_s**4 / (2. * np.pi) / (u_max**2)

    
    def plot_density_baselines(self, z):
        """
            Quick check of n_b(u)
        """
        lambda_obs = 0.21 * (1. + z)
        u_array = np.linspace(30, 60000, 1000)
        sq_array = [] # the one with N_s^2
        sq_array_d = [] # the one with N_s^4
        for u in u_array:
            sq_array.append(self.square_uni_density(u, lambda_obs, 256))
            sq_array_d.append(self.square_uni_density(u, lambda_obs, 256**2))
            
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_subplot(111)
        ax1.plot(u_array, self.number_density(u_array, lambda_obs))
        ax1.plot(u_array, sq_array, label=r'sq. uni $N_s^2$', color='purple')
        ax1.plot(u_array, sq_array_d, label=r'sq. uni $N_s^4$', color='green')
        ax1.set_xlabel(r'$u$', fontsize=14)
        ax1.set_ylabel(r'$n(u)$', fontsize=14)
        ax1.legend(loc='best')
        ax1.set_title('Baseline density distribution for PUMA', fontsize=14)
#            ax1.set_xlim(1.e-2, 1.e3)
#           ax1.set_ylim(1.e-6,1.e5)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        plt.show()
        return 1
        
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
        ang_factor = lambda_obs**2 / self.A_e # [sq. m] / [sq. m] = dimensionless
        # FOV factor
        f_factor = self.S_area / self.FOV(lambda_obs) # dimensionless
        # number density stuff
        # first get the right k perp
        kt_Mpc = k_Mpc * np.sqrt(1. - mu**2)
        u = kt_Mpc * D_c_Mpc / (2. * np.pi)
        den_factor = 2. * self.t_int * self.number_density(u, lambda_obs) # [s]
        # construct noise, note that I transform a lambda obs to km due to hubble's units
        # and I transform K into mK too
        P_N = (self.T_sys(nu_obs))**2 * D_c_Mpc**2 * (lambda_obs / 1000.) * (1. + z) / H_kms_Mpc * ang_factor**2 / den_factor * f_factor
        if self.verbose == 1:
            print('Lambda obs in km ', lambda_obs / 1000.)
            print('Nu obs in MHz ', nu_obs)
            print('Comoving distance in Mpc ', D_c_Mpc)
            print('Hubble in km/s/Mpc ', H_kms_Mpc)
            print('FOV is ', self.FOV(lambda_obs))
            print('kt in 1/Mpc is ', kt_Mpc)
            print('The u ', u)
            print('Integration time ', self.t_int)
            print('Number density for that kt ', self.number_density(u, lambda_obs))
            print('T system in mili Kelvins ', self.T_sys(nu_obs))
            prefrator = (self.T_sys(nu_obs))**2 * D_c_Mpc**2 * (lambda_obs / 1000.) * (1. + z) / H_kms_Mpc
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
            
            The inputs are redshift, wavenumber in 1/Mpc and the angle between the line of sigth mu
        """
        lambda_obs = 0.21 * (1. + z) # note that this is in meters
        nu_obs = 1420 / (1. + z) # in MHz, I got lazy so defined both hehe
        # comoving distance in Mpc
        D_c_Mpc = self.lya_c.comoving_dist(z)
        # hubble parameter in km/s/Mpc
        H_kms_Mpc = self.lya_c.hubble(z)
        # angular resolution factor
        ang_factor = lambda_obs**2 / self.A_e # [sq. m] / [sq. m] = dimensionless
        # FOV factor
        f_factor = self.S_area / self.FOV(lambda_obs) # dimensionless
        # number density stuff
        # first get the right k perp
        kt_Mpc = k_Mpc * np.sqrt(1. - mu**2)
        u = kt_Mpc * D_c_Mpc / (2. * np.pi)
        """ settle for something for this """
        N_s = 256 * 256
        den_factor = 2. * self.t_int * self.square_uni_density(u, lambda_obs, N_s) # [s]
        # construct noise, note that I transform a lambda obs to km due to hubble's units
        # and I transform K into mK too
        P_N = (self.T_sys(nu_obs))**2 * D_c_Mpc**2 * (lambda_obs / 1000.) * (1. + z) / H_kms_Mpc * ang_factor**2 / den_factor * f_factor
        return P_N

    def plot_baselines_per_dist(self):
        l_array = np.linspace(0,2100,200)
        lambda_obs = 0.21 * (5) # redshift four
        u = []
        n_sq_phys = [] # the one with N_s^2
        n_sq_phys_d = [] # the one with N_s^4
        sq_phys_sk = [] # SKA N_s^2
        sq_phys_sk_d = [] # SKA N_s^4
        for l in l_array:
            u.append(l / lambda_obs)
            n_sq_phys.append((self.square_uni_density(1., lambda_obs, 256) / lambda_obs**2) * 2. * np.pi * l)
            n_sq_phys_d.append((self.square_uni_density(1., lambda_obs, 256*256) / lambda_obs**2) * 2. * np.pi * l)
#            sq_phys_sk.append((self.square_uni_density_ska(1., lambda_obs, 911) / lambda_obs**2) * 2. * np.pi * l)
#            sq_phys_sk_d.append((self.square_uni_density_ska(1., lambda_obs, 911*911) / lambda_obs**2) * 2. * np.pi * l)
            # the u argument does not matter since uniform
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_subplot(111)
        ax1.plot(l_array, self.n_b_phys(l_array) * 2. * np.pi * l_array)
        ax1.plot(l_array, n_sq_phys, label=r'Sq. uni. approx.  $N_s^2$')
        ax1.plot(l_array, n_sq_phys_d, label=r'Sq. uni. approx.  $N_s^4$')
#        ax1.plot(l_array, sq_phys_sk, label=r'Sq. uni. approx. SKA $N_s^2$')
#        ax1.plot(l_array, sq_phys_sk_d, label=r'Sq. uni. approx. SKA $N_s^4$')
        ax1.set_xlabel(r'baseline length $\ell$ [m]', fontsize=14)
        ax1.set_ylabel(r'$n_b(\ell) \times 2\pi\ell$ [1/m]', fontsize=14)
        ax1.set_title('# of baselines per unit radial distance', fontsize=14)
        ax1.legend(loc='best')
        #            ax1.set_xlim(1.e-2, 1.e3)
        #           ax1.set_ylim(1.e-6,1.e5)
#        ax1.set_yscale('log')
#        ax1.set_xscale('log')
        plt.show()
        return 1
