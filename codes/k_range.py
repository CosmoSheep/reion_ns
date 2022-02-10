import numpy as np
import lya_convertor as lya_c

"""
    Useful code to justify the range of k chosen.
    
    Note that I included "ext-HIRAX" from https://arxiv.org/pdf/1709.07893.pdf just to confirm numbers.
"""

lya_c = lya_c.lya_convert()
# let's hardcode the minimum distances [meter], here we assume that between antenna and antenna we have roughly any distance
D_min_puma = 6
D_min_ska = 35
D_min_hirax = 10
# and the maximum distances are [meter]
D_max_puma = 2000
D_max_ska = 65000
D_max_hirax = 800
def min_max_perp(z):
    print('Redshift is ', z)
    # comoving distance in Mpc
    D_c_Mpc = lya_c.comoving_dist(z)
    # need lambda_obs
    lambda_obs = 0.21 * (1. + z)
    cte = 2. * np.pi / D_c_Mpc / lambda_obs
    print('\n order k_perp min_puma, max_puma, min_ska, max_ska, min_hirax, max_hirax \n')
    print(cte * D_min_puma, cte * D_max_puma, cte * D_min_ska, cte * D_max_ska, cte * D_min_hirax, cte * D_max_hirax)
    return cte * D_min_puma, cte * D_max_puma, cte * D_min_ska, cte * D_max_ska

min_max_perp(4.0)

min_max_perp(5.5)

# so k_perp range would be given by those numbers. So imagine a window function with this values for allowed k_perp's. However, that is not all. We need to deal with the full-k, luckily that is super easy, let's just use what other people have in the context of our two telescopes:
"""
For PUMA -- based on https://arxiv.org/pdf/1810.09572.pdf Figure 34, we will use k_min = 10^-2 h Mpc^-1. From Figure 17, same paper, we see that k_max = 1.0 h Mpc^-1 is workable but we could also move to 0.75 h Mpc^-1.
"""
k_min_puma = .01 * lya_c.h
k_max_puma = 1.0 * lya_c.h # 0.75 * lyac.h depending on what you want
print('PUMA k_mag range: '+str(k_min_puma)+'< k_mag <'+str(k_max_puma))
"""
For SKA1-Low: I think a sensible approach here is to follow 1410.7393 (Francisco's paper), from Figure 2 we can see that k_min = 0.08 h Mpc^-1 and k_max = 1 h Mpc^-1 is also a sensible choice. Obviously, one could also go for the linear argument and choose k_max = 0.2 * h Mpc^-1 but this will kill some of the EoR effects. If we want to use the linear-cut we should still push for a larger k than 0.2 h.
"""
k_min_ska = 0.08 * lya_c.h
k_max_ska = 1. * lya_c.h
print('SKA1-Low k_mag range: '+str(k_min_ska)+'< k_mag <'+str(k_max_ska))
# a couple of comments:
# In the case of PUMA note, that k_perp_min << k_min_puma, therefore k_par_min = k_min_puma.
# For SKA things are trickier since k_perp_min < k_min_ska, so one perhaps should not assume k_par_min = k_min_ska.
# I mention this because iterating over k_perps and k_parallel or k_mag and mu will be how we compute things, unless we fixed mu=0 or something.
# We could always do something fancier looking at redshift dependence but I think this is fine for our purposes. 

