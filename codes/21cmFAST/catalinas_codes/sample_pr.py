#First I call the packages that are going to be needed, there are some classes of 21cmfast that you have to call
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
with global_params.use(OMn=0.0, OMk =0.0, OMr=8.6e-5 , OMtot=1, Y_He=0.245, wl=-1.0):
    lightcone_sample = p21c.run_lightcone(
        redshift = 3.0, #minimum redshift, next time I will use 5.0
        max_redshift = 20.0, #this is the max, but you always get the data up to z~35
        lightcone_quantities=("brightness_temp", 'density', 'xH_box'), #always put the brightness_temp one, if not it doesnt works
        global_quantities=("brightness_temp", 'density', 'xH_box'),
        user_params = {"HII_DIM": 256, "BOX_LEN": 400,  "DIM":768, "N_THREADS":16  },
        cosmo_params = p21c.CosmoParams(SIGMA_8=0.81,hlittle =0.68 ,OMm = 0.31, OMb =0.04, POWER_INDEX =0.97 ),
        flag_options = {"INHOMO_RECO": True, "USE_TS_FLUCT":True },





        random_seed=12345,
        direc = '/work/catalinam/patchy_reio' #here it is where i want the boxes to be stored
    )

#I plotted the lightones combining 21cmfast plotting and matplotlib for a better resolution
plotting.lightcone_sliceplot(lightcone_sample, "density")
plt.savefig("density.png",dpi=200)
plotting.lightcone_sliceplot(lightcone_sample, "xH_box")
plt.savefig("xh_box.png",dpi=200)
plotting.lightcone_sliceplot(lightcone_sample, "brightness_temp")
plt.savefig("brightness_temp.png",dpi=200)

#I wanted to make a list of neutral hidrogen average value in terms of redshift
avg = lightcone_sample.global_xH #this gives me the neutral hydrogen values
avg1 = np.array(avg)
np.savetxt('xh_list', avg1)
z= lightcone_sample.node_redshifts #this gives me the values of redshift that go according to the global_xH
z1 = np.array(z)
np.savetxt('z_list', z1)

#Plot the global history of a neutral hydrogen from a lightcone. It uses the values above
p21c.plotting.plot_global_history(lightcone_sample, 'xH')
#plt.xlim(5.9, 15) Used to "zoom in" the plot
#plt.ylim(0, 1.0)
