import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg
import sys
import puma
from math import floor
import bias_v3
import pickle

# OM = 0.3089
H0 = 67.74 # km/s/Mpc
# h = 4.135667696e-15 # Planck constant in units of eV/Hz
# nu_0 = 13.6/h
# c = 3.0e10 # light speed in units of cm
G = 6.67e-11 # kg^-1 m^3 s^-2
# Yp = 0.249 # Helium abundance
# mH = 1.6729895e-24 # g
# bolz_k = 8.617333262145e-5 # Bolzmann constant in units of eV/K
rho_crit=3*H0**2/8/np.pi/G*H0/100 # M_solar/(Mpc)^3/h

# rhom=rho_crit*OM

# parsec=3.085677581e16 # m per parsec
# H_0=67.74 # Hubble constants now, 67.74 km/s/mpc
# G=4.30091e-9  #6.674×10−11 m3*kg−1*s−2 ### 4.30091(25)×10−3 Mpc*M_solar-1*(km/s)^2
# solar_m= 1.98847e30 #(1.98847±0.00007)×10^30 kg

# z=5.5

# Omega_m=0.3089 # Omega_m = 0.3089+-0.0062
# rhom=rho_crit*Omega_m #*(1+z)**3
# f_b=0.17 # baryon fraction


class P_21_obs:
    
    def __init__(self, k, mu, z, params, param_shift):
    
        self.k = k
        self.mu = mu
        self.z = z
        self.h = params['h']
        # self.h=0.6774
        self.Obh2 = params['Obh2']
        self.Och2 = params['Och2']
        self.mnu2 = params['mnu']
        self.As = params['As'] / 1.e9 # As read in 10^9 * As
        self.ns = params['ns']
        self.alpha_s = params['alphas']
        self.tau_re = params['taure']
#         self.OHIh2s = params['O_HIs']*0.6774**2
        self.bHI = params['bHI']
        self.OHI = params['O_HI'] / 1.e3
        # self.OHI = self.OHI(self.z)
        self.param_shift = param_shift
        self.OMh2 = self.Obh2 + self.Och2
        self.pk()
        
    # def OHI(self,z):
    #     if 3.49 < z < 4.45:
    #         return 1.18e-3
    #     elif 4.44 < z < 5.51:
    #         return 0.98e-3

    def Tb_mean(self):
        '''
        output in units of mK
        '''
        
        return 27*np.sqrt((1+self.z)/10*0.15/self.OMh2)*(self.OHI*self.h**2/0.023)


    def pk(self):
        '''
        output is a function P(k), k in units of Mpc^{-1}, P(k) is in units of Mpc^3
        need to determine file names
        '''

        pk = []
        zs = np.arange(3.5,6.0,0.5)
        for z in range(5):
            # z from 5.5 to 3.5, files numbered 1-5
            data=np.loadtxt('./patchy_reion_ns/%s/z%d_pk.dat' % (self.param_shift,-z+5))
            pk.append(data[:,1])
        k = data[:,0]
        pk = np.array(pk) / self.h**3
#         pk *= np.array(k)**3/2/np.pi**2 
        coords = []
        for i in zs:
            for j in k:
                coords.append((i,j * self.h))
            
        self.pk_fun = interpolate.LinearNDInterpolator(coords, pk.flatten())

    def f(self):
    # Growth rate = OM(z)^0.545 = (OM*(1+z)^3/(OM*(1+z)^3+(1-OM)))^0.545 ref: 1709.07893
        OM = (self.Obh2+self.Och2)/self.h**2
        return (OM*(1+self.z)**3/(OM*(1+self.z)**3+1-OM))**0.545


    def P_fid(self):
        '''
        Output units mK^2
        '''
        return self.Tb_mean()**2 * (self.bHI + self.mu**2*self.f())**2 * self.pk_fun(self.z,self.k)
    
    def PN(self,t_int,b):
        '''
        Input: t_int is 21cm IM mission time in units of hour, b is the bandwidth in MHz
        Output units mK^2 Mpc^3
        '''
#         t_int = 24*365*5
#         b = 8
        OM = self.OMh2/self.h**2
        Or = 0
        noise = puma.puma(t_int, b, OM, self.h,Or)
        PN = noise.noise_power_Mpc(self.z,self.k,self.mu)
        
        return PN
    
    def P_21obs(self):
        P_fid = self.P_fid()+self.PN(24*365*5,3)
        print('At k = %f, mu = %f, z=%f, param_scenario = %s, P_fid = %f '% (self.k, self.mu, self.z, self.param_shift, P_fid))

        return P_fid
    

class P_patchy_reion:
    
    def __init__(self, k, mu, z, params, verbose = True):
    
        self.k = k
        self.mu = mu
        self.z = z
        self.h = params['h']
        # self.h = 0.6774
        self.Obh2 = params['Obh2']
        self.Och2 = params['Och2']
        self.mnu2 = params['mnu']
        self.As = params['As'] / 1.e9
        self.ns = params['ns']
        self.alpha_s = params['alphas']
        self.tau_re = params['taure']
        self.bHI = params['bHI']
        self.OHI = params['O_HI'] / 1.e3
        # self.OHI = self.OHI(self.z)
        self.verbose = verbose

        self.OMh2 = self.Obh2+self.Och2
        

        # self.param_shift = param_shift

        self.delta_crit=1.686 # critcal overdensity for collapse
        self.Mhalos=np.logspace(7,16,10000)
        self.ks=np.logspace(-4,3,1000)
        
        # G=4.30091e-9
        # rho_crit=3*(self.h*100)**2/8/np.pi/G*self.h # M_solar/(Mpc)^3/h
        # self.rhom=rho_crit*self.OMh2/self.h**2 #*(1+z)**3
        
#         print(self.rhom)

        
    # def OHI(self,z):
    #     if 3.49 < z < 4.45:
    #         return 1.18e-3
    #     elif 4.44 < z < 5.51:
    #         return 0.98e-3
        
    def Tb_mean(self):
        
        return 27*np.sqrt((1+self.z)/10*0.15/self.OMh2)*(self.OHI*self.h**2/0.023)

    def reion_his(self,filename):
        data = np.loadtxt(filename)
        z_rh =data.T[0]
        xH = data.T[1]
        return interpolate.interp1d(z_rh,xH)
        

    def Pk(self):
        '''
        output is a function P(k), k in units of Mpc^{-1}, P(k) is in units of Mpc^3
        need to determine file names
        '''

        pk = []
        zs = np.arange(3.5,6.0,0.5)
        for z in range(5):
            # z from 5.5 to 3.5, files numbered 1-5
            data=np.loadtxt('./patchy_reion_ns/fid/z%d_pk.dat' % (-z+5))
            pk.append(data[:,1])
        k = data[:,0]
        pk = np.array(pk) / self.h**3
#         pk *= np.array(k)**3/2/np.pi**2 
        coords = []
        for i in zs:
            for j in k:
                coords.append((i,j* self.h))
            
        self.pk_fun = interpolate.LinearNDInterpolator(coords, pk.flatten())
        return self.pk_fun


#     def halo_func(self):
#         pk_fun=self.Pk()

#         A=0.186
#         a=1.47
#         b=2.57
#         c=1.19
        
#         sigmas=[]
# #         print(pk_fun(self.z,self.ks))
        
#         for m in self.Mhalos:
#             R=(3.*m/4./np.pi/self.rhom)**(1./3)
#             w_kR=3./(self.ks*R)**3*(np.sin(self.ks*R)-self.ks*R*np.cos(self.ks*R))
#             d_sigma2=pk_fun(self.z,self.ks)*w_kR**2*self.ks**2/(2*np.pi**2)
# #             print(d_sigma2)
#             sigma_2=integrate.simps(d_sigma2,self.ks)
# #             print(sigma_2)
#             sigma=np.sqrt(sigma_2)  
#             sigmas.append(sigma)

#         sigmas=np.array(sigmas)
#         self.sigmas=sigmas
#         f_sigma=A*((sigmas/b)**(-a)+1)*np.exp(-c/sigmas**2)

#         self.dndM=f_sigma*self.rhom/self.Mhalos*np.gradient(np.log(sigmas**(-1)),self.Mhalos)

    def f(self):
    # Growth rate = OM(z)^0.545 = (OM*(1+z)^3/(OM*(1+z)^3+(1-OM)))^0.545 ref: 1709.07893
        OM = self.OMh2/self.h**2

        return (OM*(1+self.z)**3/(OM*(1+self.z)**3+1-OM))**0.545
        
#     def M_b(self,Mhalo,z_re,file,xi_arr):
#         f_b = 0.1573
#         xi_arr = self.reion_his('xH_concat_fc_fid')
#         MF=pd.read_csv('fmass_mean.csv',index_col='z_obs')
#         z_res=[6.0,7.0,8.0,8.5,9.0,10.0,11.0,12.0]

#         if z_re==6.0: 
#             Mb = f_b*Mhalo*(1+(2**(1./3)-1)*MF.loc[self.z_obs][0]/Mhalo)**(-3.)
#         else: 
#             z_range=np.arange(6.0,z_re+0.01,0.01)
#             xi_dot=np.gradient(xi_arr(z_range),z_range)
#             mf = interpolate.interp1d(np.array(z_res),MF.loc[self.z_obs].to_numpy(),fill_value="extrapolate")
#             Mb = -integrate.simps(f_b*Mhalo*(1+(2**(1./3)-1)*mf(z_range)/Mhalo)**(-3.)*xi_dot,z_range)

#         return Mb

#     def rho_HI(self):
#         self.halo_func()
#         xi_arr = self.reion_his('xH_concat_fc_fid')
#         f_b = 0.1573

#         rho_HI=[]
#         z_res=[6.0,7.0,8.0,8.5,9.0,10.0,11.0,12.0]
#         z_obs = np.arange(3.5,5.6,0.1)
        
#         coords = []
        
#         for z_re in z_res:
#             for zobs in z_obs:
#                 M_baryon_sv = np.array([self.M_b(mh,z_re,file_sv,xi_sv)[0] for mh in self.Mhalos])
#                 mf = MF.loc[round(zobs,1)][str(z_re)]
#                 M_baryon = f_b*self.Mhalos*(1+(2**(1./3)-1)*mf/self.Mhalos)**(-3.)
# #                 Mb = np.array([integrate.simps(f_b*Mhalo*(1+(2**(1./3)-1)*mf(z_range)/Mhalo)**(-3.)*xi_dot,z_range) for Mhalo in self.Mhalos])
#                 rho_HI.append(integrate.simps(self.dndM*M_baryon,self.Mhalos))
#                 coords.append((z_re,zobs))
        
#         rho_HI = interpolate.LinearNDInterpolator(coords, np.array(rho_HI).flatten())
#         return rho_HI
    
    def rho_HI(self):
    #     z_res=[6.0,7.0,8.0,8.5,9.0,10.0,11.0,12.0]
    #     z_obs = np.arange(3.5,5.6,0.1)

    #     coords = []
        
    #     for z_re in z_res:
    #         for zobs in z_obs:               
    #             xi = bias_v3.b_sink(z_re, zobs, 'fid')
    #             xi_sv, xi_nv = xi.xi_func()
    #             rho = bias_v3.bias_v(z_re, zobs, 'fid')
    #             M_baryon = np.array([rho.M_b(mh,z_re,'fmass_mean.csv',xi_sv)[0] for mh in self.Mhalos])
    #             rho_HI.append(integrate.simps(self.dndM*M_baryon,self.Mhalos))
    #             coords.append((z_re,zobs))

        with open('./rho_HI_func.pkl','rb') as f:
           rho_HI = pickle.load(f)
           
        return rho_HI

    def reion_mid(self):
        xi_arr = self.reion_his('xH_concat_fc_fid')
        for z in np.arange(8.0,6.0,-0.01):
            if (np.abs(xi_arr(z)-0.5) < 0.001): break
        return z

    def dpsi_dz(self):
        zmin = 6.0
        zmax = 12.0
        z_res = np.arange(zmin,zmax+0.1,0.01)
        z_re_mean = self.reion_mid()
#         print(z_re_mean)
        
        
        rho_HI = self.rho_HI()
#         print(rho_HI(z_res,self.z),rho_HI(z_re_mean,self.z))
    
        dpsi_dz = np.gradient(np.log(rho_HI(z_res,self.z)/rho_HI(z_re_mean,self.z)),z_res)
        dpsi_dz[np.isnan(dpsi_dz)]=0

        return dpsi_dz

    def PmxH(self):
        zmin = 6.0
        zmax = 12.0
        z_res = np.arange(zmin,zmax+0.1,0.01)
        Pc = np.loadtxt('matter_cross_HI_21cmfastv2_fid.txt')

        # original Pk in dimensionless, transform to Mpc^3
        Pk = Pc.T[2] * 2 * np.pi**2 / Pc.T[1]**3
        # Pk = Pc.T[2]

        # interpolate Pk on (z,k) plane
        P_m_xH_func = interpolate.LinearNDInterpolator(Pc[:,0:2], np.array(Pk).flatten())
        
        PmxH = P_m_xH_func(z_res,self.k)

        return PmxH

    def P_m_psi(self):
        zmin = 6.0
        zmax = 12.0
        z_res = np.arange(zmin,zmax+0.1,0.01)
 
        dpsi_dz = self.dpsi_dz()

        PmxH = self.PmxH()
        
        P_m_psi = -integrate.simps(dpsi_dz*PmxH*(np.ones(z_res.shape)+z_res)/(1+self.z),z_res) # D prop to a
        
        return P_m_psi
        
    def P_reion(self):

        P_patchy = 2 * self.Tb_mean()**2 * (self.bHI + self.mu**2 * self.f()) * self.P_m_psi()

        if self.verbose is True:
            print('At k = %f, mu = %f, z=%f, P_patchy = %f '% (self.k, self.mu, self.z, P_patchy))

        return P_patchy


class Fisher:

    def __init__(self,z,nk,nmu,dz,kmax = 0.4, verbose = True):

        self.z = z
        self.dz = dz
        self.kmax = kmax
        self.nk = nk
        self.nmu = nmu
        self.verbose = verbose
        

        self.params={}
        self.params['h'] = 0.6774
        # self.h = 0.6774
        self.h = self.params['h']
        self.params['Obh2'] = 0.02230
        self.params['Och2'] = 0.1188
        self.params['mnu'] = 0.194
        self.params['As'] = 2.142 # 10^9 * As
        self.params['ns'] = 0.9667
        self.params['alphas'] = -0.002
        self.params['taure'] = 0.066
        self.params['bHI'] = float(self.bHI(self.z))
        self.params['O_HI'] = self.OHI(self.z) * 1.e3
        self.OM = (self.params['Obh2']+self.params['Och2'])/self.h**2

        zbin_width = self.DA(self.z + self.dz) - self.DA(self.z - self.dz)
        self.kmin = 2 * np.pi / zbin_width

        self.F = None

        self.Finv = None

    def OHI(self,z):
        if 3.49 < z < 4.5:
            return 1.18e-3
        elif 4.5 <= z < 5.51:
            return 0.98e-3

    # def bHI(self,z):
    #     bHI = np.array([2.2578, 2.3970, 2.5314, 2.6581, 2.7739])                           
    #     zs = np.array([3.5, 4.0, 4.5, 5.0, 5.5])
    #     bHIs = interpolate.interp1d(zs,bHI)

    #     return bHIs(z)

    def bHI(self,z):
        # 1804.09180, Table 5
        if 3.49 < z < 4.5:
            bHI = 2.82
        elif 4.5 <= z <5.51:
            bHI = 3.18

        return bHI

    def DA(self,z):
        zs = np.linspace(0,z,1000,endpoint=True)
        chi = 3.e5/(100*self.h)*integrate.simps(1./np.sqrt(1-self.OM+self.OM*(1+zs)**3),zs)

        return chi

    def V(self,z):
        '''
        Survey volume for PUMA, covering half sky
        '''	

        f_sky = 0.5
        A = 4*np.pi*f_sky # in units of sr
        v = A / 3 * self.DA(z)**3

        return v

    def dparams(self,k,mu,z):

        dP_dparam = []
        dparams={}
        dparams['h'] = 0.007
        dparams['Obh2'] = 0.0003
        dparams['Och2'] = 0.002
        dparams['mnu'] = 0.004
        dparams['As'] = 0.02 # 10^9 * dAs
        dparams['ns'] = 0.012
        dparams['alphas'] = 0.00002
        dparams['taure'] = 0.002

        dparams['bHI'] = 0.03
        dparams['O_HI'] = 0.01

        for i,par in enumerate(dparams.keys()):
            params_p = self.params.copy()
            params_m = self.params.copy()

            params_p[par] +=  dparams[par]
            params_m[par] -=  dparams[par]

            if par == 'bHI' or par == 'O_HI':
                P_fid_plus = P_21_obs(k,mu,z,params_p,'fid').P_21obs()
                P_fid_minus = P_21_obs(k,mu,z,params_m,'fid').P_21obs()

            else:
                P_fid_plus = P_21_obs(k,mu,z,params_p,par+'p').P_21obs()
                P_fid_minus = P_21_obs(k,mu,z,params_m,par+'m').P_21obs()

            # if par == 'As': 
            #     dparams[par] /= 1.e9

            # elif par == 'O_HI':
            #     dparams[par] /= 1.e3

            dP_dparam.append((P_fid_plus-P_fid_minus)/2/dparams[par])

        return dP_dparam

    def Cov_inv(self,k,mu,z, dk, dmu):

        nmodes = k**2 / (2 * np.pi)**2 * dk * dmu * (self.V(z+self.dz/2) - self.V(z-self.dz/2))
        P_fid = P_21_obs(k,mu,z,self.params,'fid').P_21obs()

        return nmodes / 2. / P_fid**2

    def Cov_inv_arr(self):
        ks0 = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.nk, endpoint = True)
        dks = np.diff(ks0)
        mus = np.linspace(-1, 1., self.nmu+1, endpoint = True)
        mus = (mus[1:] + mus[:-1]) / 2
        ks = 10**((np.log10(ks0[1:]) + np.log10(ks0[:-1])) / 2)

        dmu = 2 / self.nmu

        # generate and save 2*Nmode/Pk**2 for the sake of time
        zs = np.linspace(zmin, zmax, int((zmax - zmin)/dz)+1, endpoint=True)
        zs = (zs[1:] + zs[:-1])/2

        Cov_inv_arr = []
        for z in zs:
            this_arr = []
            for i in range(self.nk-1):
                for mu in mus:
                    k = ks[i]
                    dk = dks[i]
                    mu_min = self.kmin / k
                    if np.abs(mu) < mu_min: continue
                    if self.verbose is True:
                        print('mu = %f' % mu)
                    # k = self.kmin + self.dk / 2 + i * self.dk
                    # mu = -1. + self.dmu / 2 + j * self.dmu
                    this_arr.append(self.Cov_inv(k, mu, z, dk, dmu))
            Cov_inv_arr.append(this_arr)

        self.Cov_inv_arr = np.array(Cov_inv_arr)
        return self.Cov_inv_arr

    def Fmat(self,k,mu,z, dk, dmu, Cov):
        # zmax = 5.5
        # zmin = 3.5

        # P_reion = P_patchy_reion(k,mu,z,self.params).P_reion()

        F = np.zeros((len(self.params),len(self.params)))
        dP_dparam = self.dparams(k,mu,z)

        # Cov = self.Cov_inv(k, mu, z, dk, dmu)

        for i in range(len(self.params)):
            for j in range(len(self.params)):
                F[i,j] = Cov * dP_dparam[i] * dP_dparam[j]

        return F

    def get_F(self):

        # nk =floor((self.kmax-self.kmin)/self.dk)
        # nmu = floor(2/self.dmu)

        ks0 = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.nk, endpoint = True)
        dks = np.diff(ks0)
        mus = np.linspace(-1, 1., self.nmu+1, endpoint = True)
        mus = (mus[1:] + mus[:-1]) / 2
        ks = 10**((np.log10(ks0[1:]) + np.log10(ks0[:-1])) / 2)

        dmu = 2 / self.nmu
        if self.Cov_inv_arr is None:
            self.Cov_inv_arr()

        self.F = np.zeros((len(self.params),len(self.params)))

        z_index = int((self.z - 3.5)/0.2)
        Cov = self.Cov_inv_arr()
        count = 0
        for i in range(self.nk-1):
            for mu in mus:
                k = ks[i]
                dk = dks[i]
                mu_min = self.kmin / k
                if np.abs(mu) < mu_min: continue
                if self.verbose is True:
                    print('mu = %f' % mu)                
                self.F += self.Fmat(k,mu,self.z, dk, dmu, Cov[z_index][count]) 
                count += 1


        return self.F

    def param_mean(self, param_num, data):
    
        return np.sum(data.T[0]*data.T[param_num])/np.sum(data.T[0])

    def param_cov(self, param_nums, data):

        param1 = param_nums[0]
        param2 = param_nums[1]
        param1_mean = self.param_mean(param1, data)
        param2_mean = self.param_mean(param2, data)
        
        return np.sum(data.T[0]*(data.T[param1]-param1_mean)*(data.T[param2]-param2_mean))/np.sum(data.T[0])


    def CMB_cov(self, file_name):

        data = np.loadtxt(file_name)

        param_planck = {}
        param_planck['Obh2'] = 2
        param_planck['Och2'] = 3
        param_planck['taure'] = 5
        param_planck['mnu'] = 6
        param_planck['ns'] = 8
        param_planck['H0'] = 30
        param_planck['10^9As'] = 44

        cov_planck = np.zeros((7,7))
        for i,par1 in enumerate(param_planck.keys()):
                for j,par2 in enumerate(param_planck.keys()):
                    param1 = param_planck[par1]
                    param2 = param_planck[par2]
                    cov_planck[i,j] += self.param_cov([param1,param2],data)
                #     if j==5: cov_planck[i,j] /= 100
                #     # if j==5: cov_planck[i,j] /= 1.e9
                # if i==5: cov_planck[i,j] /= 100
                # # if i==5: cov_planck[i,j] /= 1.e9
        cov_planck[5,:] /= 100
        cov_planck[:,5] /= 100

        return cov_planck

    def add_prior(self, cov_prior):
        '''
        Add CMB priors to 21cm Fisher matrix
        '''

        # 21cm Fisher matrix parameter index mapping
        param_map = {}
        param_map['Obh2'] = 1
        param_map['Och2'] = 2
        param_map['taure'] = 7
        param_map['mnu'] = 3
        param_map['ns'] = 5
        param_map['H0'] = 0
        param_map['As'] = 4

        F_CMB = np.linalg.inv(cov_prior)
        F_prior = self.F.copy()

        for i,par1 in enumerate(param_map.keys()):
            for j,par2 in enumerate(param_map.keys()):
                pos1 = param_map[par1]
                pos2 = param_map[par2]
                F_prior[pos1,pos2] += F_CMB[i,j]

        return F_prior

    def get_Finv(self):
        if self.F is None:
            print('Generate Fishser matrix\n')
            self.F = self.get_F()

        else:
            self.cov_prior = self.CMB_cov('/Users/heyang/Downloads/COM_CosmoParams_fullGrid_R3/base_mnu/plikHM_TTTEEE_lowl_lowE_lensing/base_mnu_plikHM_TTTEEE_lowl_lowE_lensing_2.txt')
            F_prior = self.add_prior(self.cov_prior)
            self.Finv = np.linalg.inv(F_prior)
        
        return self.Finv

    def this_param_shift(self,z, k,mu, dk, dmu, param, Cov):
        params = self.params.copy()
        params['bHI'] = float(self.bHI(z))
        params['O_HI'] = self.OHI(z) * 1.e3

        P_reion = P_patchy_reion(k,mu,z,params).P_reion()
        d_param = 0

        self.dP_dparam = self.dparams(k,mu,z)

        if self.Finv is None:
            raise Exception("Need an Finv to continue")
            # self.get_Finv()

        param_index = list(params.keys()).index(param)


        for j in range(8):

            d_param += self.Finv[param_index,j] * Cov * P_reion * self.dP_dparam[j]

        return d_param

    def param_shift(self, zmin, zmax, dz, nk, nmu, param):
        shift = 0

        zs = np.linspace(zmin, zmax, int((zmax - zmin)/dz)+1, endpoint=True)
        zs = (zs[1:] + zs[:-1])/2

        for z in zs:
            kmin = 2 * np.pi / (self.DA(z + dz/2) - self.DA(z - dz/2))
            ks = np.logspace(np.log10(kmin), np.log10(self.kmax), nk+1, endpoint = True)
            dks = np.diff(ks)
            ks = 10**((np.log10(ks[1:]) + np.log10(ks[:-1])) / 2)
            mus = np.linspace(-1., 1., nmu+1)
            mus = (mus[1:] + mus[:-1]) / 2
            dmu = 2. / nmu

            z_index = int((z - 3.5)/0.2)
            Cov = self.Cov_inv_arr()
            count = 0
            for i in range(nk):
                k = ks[i]
                dk = dks[i]
                mu_min = kmin / k
                for mu in mus:
                    if np.abs(mu) < mu_min: continue
                    shift += self.this_param_shift(z, k, mu, dk, dmu, param, Cov[z_index][count])
                    count += 1

        return shift


    def ns_shift(self,k,mu):
        P_reion = P_patchy_reion(k,mu,self.z,self.params).P_reion()
        d_ns = 0

        self.dP_dparam = self.dparams(k,mu,self.z)

        if self.Finv is None:
            self.get_Finv()

        for j in range(len(self.params)):
            d_ns += self.Finv[5,j]*self.Cov_inv(k,mu,self.z)*P_reion*self.dP_dparam[j]

        return d_ns


