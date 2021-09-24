import py21cmfast as p21c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from py21cmfast import plotting
from py21cmfast import cache_tools
from py21cmfast import global_params

#Just to make sure its in the right version
print(f"Using 21cmFAST version {p21c.__version__}")

#Here I produce the neutral hydrogen and density boxes, using the parameters needed
#Global parameters always have to be stated this way, the other ones(cosmo,astro,user) go below run_lightone
with global_params.use(Pop2_ion=4800,OMn=0.0, OMk =0.0, OMr=8.6e-5 , OMtot=1, Y_He=0.245, wl=-1.0, SMOOTH_EVOLVED_DENSITY_FIELD =1, R_smooth_density=0.2, HII_ROUND_ERR= 1e-5, N_POISSON=-1 , MAX_DVDR=0.2,DELTA_R_FACTOR=1.1, DELTA_R_HII_FACTOR=1.1, OPTIMIZE_MIN_MASS=1e11, SHETH_b=0.15, SHETH_c=0.05, ZPRIME_STEP_FACTOR=1.02 ):
    lightcone_sample_4800 = p21c.run_lightcone(
        redshift = 5.0, #minimum redshift, next time I will use 5.0
        max_redshift = 15.0, #this is the max, but you always get the data up to z~35
        lightcone_quantities=("brightness_temp", 'density', 'xH_box'), #always put the brightness_temp one, if not it doesnt works
        global_quantities=("brightness_temp", 'density', 'xH_box'),
        user_params = {"HII_DIM": 256, "BOX_LEN": 400,  "DIM":768, "N_THREADS":16  },
        cosmo_params = p21c.CosmoParams(SIGMA_8=0.81,hlittle =0.68 ,OMm = 0.31, OMb =0.04, POWER_INDEX =0.97 ),
        astro_params = {'R_BUBBLE_MAX':50, 'L_X':40.5},
        flag_options = {"INHOMO_RECO": True, "USE_TS_FLUCT":True, "USE_MASS_DEPENDENT_ZETA":True },





        random_seed=12345,
        direc = '/work/catalinam/patchy_reio/fourth_sample' #here it is where i want the boxes to be stored
    )

#I plotted the lightones combining 21cmfast plotting and matplotlib for a better resolution
#plotting.lightcone_sliceplot(lightcone_sample, "density")
#plt.savefig("density.png",dpi=200)
plotting.lightcone_sliceplot(lightcone_sample_4800, "xH_box")
plt.savefig("xh_box_4800.png",dpi=200)
#plotting.lightcone_sliceplot(lightcone_sample, "brightness_temp")
#plt.savefig("brightness_temp.png",dpi=200)

#I wanted to make a list of neutral hidrogen average value in terms of redshift
avg = lightcone_sample_4800.global_xH #this gives me the neutral hydrogen values
#avg1 = np.array(avg)
#np.savetxt('xh_list', avg1)
z= lightcone_sample_4800.node_redshifts #this gives me the values of redshift that go according to the global_xH
#z1 = np.array(z)
#np.savetxt('z_list', z1)
file_new = pd.DataFrame({'Redshift':z, 'xH_avg_value':avg})
new = file_new.style.hide_index()
file_new.to_csv('z_and_xH_values_4800',index=False)

#Plot the global history of a neutral hydrogen from a lightcone. It uses the values above
p21c.plotting.plot_global_history(lightcone_sample_4800, 'xH')
plt.xlim(5.9, 15)# Used to "zoom in" the plot
plt.savefig('global_history_4800.png', dpi=200)
#plt.xlim(5.9, 15) Used to "zoom in" the plot
