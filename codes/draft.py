%matplotlib inline
import matplotlib.pyplot as plt
import os
# We change the default level of the logger so that
# we can see what's happening with caching.
import logging, sys, os
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)
import py21cmfast as p21c
# For plotting the cubes, we use the plotting submodule:
from py21cmfast import plotting
# For interacting with the cache
from py21cmfast import cache_tools

print(f"Using 21cmFAST version {p21c.__version__}")

if not os.path.exists('/users/PCON0003/osu10670/Software/21cmFAST_py/cache'):
	os.mkdir('/users/PCON0003/osu10670/Software/21cmFAST_py/cache')

p21c.config['direc'] = '/users/PCON0003/osu10670/Software/21cmFAST_py/cache'
cache_tools.clear_cache(direc="/users/PCON0003/osu10670/Software/21cmFAST_py/cache")

initial_conditions = p21c.initial_conditions(
user_params = {"HII_DIM": 200, "BOX_LEN": 300, "DIM": 800, "N_THREADS": 8},
cosmo_params = p21c.CosmoParams(SIGMA_8=0.8),
flag_options = {"INHOMO_RECO": True, "USE_HALO_FIELD": False},
astro_params = {"HII_EFF_FACTOR": 25.0 },
global_params = {"N_POISSON": -1, "ALPHA_UV": 5, "SMOOTH_EVOLVED_DENSITY_FIELD": 1, "R_smooth_density": 0.2, "SECOND_ORDER_LPT_CORRECTIONS": True, "SECOND_ORDER_LPT_CORRECTIONS":1, "HII_ROUND_ERR": 1e-5, "FIND_BUBBLE_ALGORITHM": 2, },
random_seed=54321
)

#os.system('21cmfast init --config=/users/PCON0003/osu10670/Software/21cmFAST_py/user_data/test.yml')
#os.system('21cmfast perturb redshift=8.0')



########  Memory Usage Check  ########
memusage -T ./drive_logZscroll_Ts

#######   xH Plot   ########
import os
import matplotlib.pyplot as plt

z = []
xH = []

for file in os.listdir('.'):
	if file.startswith('xH'):
		z.append(float(file[12:18]))
		xH.append(float(file[21:29]))

plt.plot(z,xH,marker='x')
plt.xlim([5.9,15])
plt.xlabel('Redshift')
plt.ylabel('Neutral fraction')
plt.grid(True)
plt.show()


############### MAKE .gif file ################
import os
import imageio

filenames=os.listdir('./pk_fig/')
filenames.sort(reverse=True)
images = []
for filename in filenames:
	if filename.endswith('.png'):
		images.append(imageio.imread('./pk_fig/'+filename))

imageio.mimsave('./pk_fig/ps.gif', images)




############# sliceplot.py ################
ml sw python/3.6-conda5.2 python/2.7-conda5.2

python sliceplot.py -i ../Boxes/xH_nohalos_z*






################# plot power spectrum density ###################
import os
import matplotlib.pyplot as plt

filenames=os.listdir('./')
filenames.remove('pk_fig')
for file in filenames:
	k=[]
	pk=[]
	with open(file, 'r') as f:
		if file.endswith('_v3'):
			data=f.read()
	data=data.split('\n')
	for i in data[:-1]:
		k.append(float(i.split('\t')[0]))
		pk.append(float(i.split('\t')[1]))
	plt.scatter(k,pk,label='z='+file[4:10],c='black')
	plt.legend()
	plt.xlabel('k')
	plt.ylabel('Power density')
	plt.savefig('./pk_fig/'+file+'.png')
	plt.clf()



#################### Ts, Tk, Tcmb plots ########################
import matplotlib.pyplot as plt
with open('global_evolution_zetaIon25.00_Nsteps40_zprimestepfactor1.020_L_X3.2e+40_alphaX1.0_TvirminX0.0e+00_Pop2_200_300Mpc', 'r') as f:
	data=f.read()

data=data.split('\n')
z=[]
Tk=[]
Ts=[]
Tcmb=[]
for i in data[:-1]:
	z.append(float(i.split('\t')[0]))
	Tk.append(float(i.split('\t')[2]))
	Ts.append(float(i.split('\t')[4]))
	Tcmb.append(float(i.split('\t')[5]))

plt.plot(z,Tk,z,Ts,z,Tcmb)
plt.legend(['Tk','Ts','Tcmb'])
plt.xlabel('z')
plt.ylabel('T (K)')
plt.yscale('log')
plt.show()


############# open binary files ##################
import numpy as np

def load_binary_data(filename, dtype=np.float32): 
     """ 
     We assume that the data was written 
     with write_binary_data() (little endian). 
     """ 
     f = open(filename, "rb") 
     data = f.read() 
     f.close() 
     _data = np.fromstring(data, dtype) 
     if sys.byteorder == 'big':
       _data = _data.byteswap()
     return _data 


#### Generate delta_xh power spectrum using delta_xh_ps.c ############
import os
import sys

files=os.listdir('./')

delta_files=[]
xh_files=[]
delxh_ps=[]
for file in files:
	if file.startswith('xH_nohalos_z'):
			xh_files.append(file)
			z=file[12:18]
			delta_files.append('updated_smoothed_deltax_z'+z+'_200_300Mpc')
			delxh_ps.append('delxh_ps_z'+z)

xh_files.sort()
delta_files.sort()
delxh_ps.sort()

for i in range(len(xh_files)):
	print('at z='+xh_files[i][12:18])
	os.system('../Programs/delta_xh_ps '+xh_files[i]+' '+delta_files[i]+' ../Output_files/delxh_power_spec/'+delxh_ps[i]+' '+xh_files[i][21:29])


################# plot delta_xh power spectrum ###################
import os
import matplotlib.pyplot as plt
import numpy as np

filenames=os.listdir('./')

for i in range(5,20):
	z=[]
	pk=[]
	for file in filenames:
		if file.endswith('swp'): continue
		with open(file, 'r') as f:
			data=f.read()
		
		data=data.split('\n')
		z.append(float(file[-6:]))
		pk.append(float(data[i].split('\t')[1]))
	pk=np.nan_to_num(pk)
	plt.plot(z[:45],pk[:45],'x-',linewidth=0.7,label='k='+data[i].split('\t')[0])

plt.legend()
plt.xlabel('z')
plt.ylabel(r'$(k^3/2{\pi}^2)P_{m,x_{HI}}$')
plt.show()



##############################################
#######       Plot MF - z_re curves        #######
##############################################
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

z_res = ['6.0','7.0','8.0','8.5','9.0','10.0','11.0','12.0']
colors = cm.Set2(np.linspace(0,1,len(z_res)))

df = pd.read_csv('fmass_mean.csv')
df_nv = pd.read_csv('fmass_mean-nv.csv')
df_err = pd.read_csv('fmass_std.csv')
df_err_nv = pd.read_csv('fmass_std-nv.csv')


	
for j in range(10):
	mf=[]
	for i in range(len(z_res)):
		mf.append(df[z_res[i]][j])
	plt.plot([float(z) for z in z_res], mf, linewidth=0.8, label=r'$z_{obs}=$'+str(df['z_obs'][j]))

z_obs = df['z_obs'].to_numpy()

for i,color in enumerate(colors):
	mf = df[z_res[i]].to_numpy()
	err = df_err[z_res[i]].to_numpy()
#	plt.plot(z_obs,mf,label=r'$z_{re}$='+z_re)
	plt.errorbar(z_obs,mf,err,linewidth=0.8, color=color,label=r'$z_{re}$='+z_res[i])

plt.legend()
plt.xlabel(r'$z_{obs}$')
plt.ylabel(r'Filtering Mass ($M_{\bigodot}/h$)')
plt.show()


for i,color in enumerate(colors):
	mf = df[z_res[i]].to_numpy()/df_nv[z_res[i]].to_numpy()
	plt.plot(z_obs,np.log(mf),linewidth=0.8, color=color, label=r'$z_{re}$='+z_res[i])

# plt.yscale('log')
plt.legend()
plt.xlabel(r'$z_{obs}$')
plt.ylabel(r'Filtering Mass Ratio $MF_{sv}/MF_{nv}$')
plt.show()


