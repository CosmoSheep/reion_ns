import numpy as np
from scipy.interpolate import interp2d
from classy import Class
import matplotlib.pyplot as plt

cosmo_params = {
    'output': 'mPk',
    'sigma8': 0.8159,
    'n_s': 0.9667,
    'h': 0.6774,
    'Omega_b': 0.0486,
    'Omega_cdm': 0.3088,
    'z_reio': 7.93,
    'non linear': 'halofit',
    'halofit_min_k_max': 10,
    'format': 'CLASS',
    'z_max_pk': 35.0
}
cosmo = Class()
cosmo.set(cosmo_params)
cosmo.compute()
h = cosmo.h()

def P_m_Mpc(k_hMpc, z):
    """
    Returns the 3D matter power spectrum obtained from CLASS in units of Mpc^3 (no little h!). Note that this is a function of redshift too.
    
    Inputs: k [h Mpc^-1], z
    
    Outputs: P_m_CDM [Mpc^3]
    """
    return cosmo.pk_lin(k_hMpc * h, z)

#fast_file = 'cross_cropped_power_fc_fid.txt' cropped is not good
fast_file = 'cross_power_concat_fc_fid.txt'
z_re, k_Mpc_fast, cross = np.loadtxt(fast_file, unpack=True)
z_re = np.unique(z_re)
k_Mpc_fast = np.unique(k_Mpc_fast)
P_cross = interp2d(z_re, k_Mpc_fast, cross)
k_min = k_Mpc_fast[0]
k_max = k_Mpc_fast[-1]

def cross_power(k_Mpc, z):
    if k_Mpc <= k_max and k_Mpc >= k_min:
        return P_cross(z, k_Mpc)
    elif k_Mpc > k_max:
        return 0
    elif k_Mpc < k_min:
        bias_scaling = P_cross(z, k_min) / P_m_Mpc(k_min / h, z)
#        print('bias ', bias_scaling)
#        print('matter ', P_m_Mpc(k_Mpc / h, z))
#        print('cross ', bias_scaling * P_m_Mpc(k_Mpc / h, z))
        return bias_scaling * P_m_Mpc(k_Mpc / h, z)

k_plot = np.logspace(-4,1,1000)
z_test = 7.69
cross_plot = np.zeros(len(k_plot))
data_plot = np.zeros(len(k_Mpc_fast))
for i in range(0,len(k_plot)):
    cross_plot[i] = cross_power(k_plot[i], z_test)

for i in range(0,len(k_Mpc_fast)):
    data_plot[i] = P_cross(z_test, k_Mpc_fast[i])

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
ax1.plot(k_plot, cross_plot)
ax1.plot(k_Mpc_fast, data_plot)
ax1.set_xlabel(r'k  [Mpc$^{-1}$]', fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}(k,z)$', fontsize=14)
ax1.set_xscale('log')
plt.show()

np.savetxt('matter_cross_HI_21cmfastv2_fid.txt', np.transpose([k_plot, cross_plot]), fmt='%e', delimiter=' ')
