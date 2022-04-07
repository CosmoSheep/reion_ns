import os
import fileinput
# just a script to run the extrapolation code
# unfortunately I already edited the other one so instead of iterating files which would have been faster, I did this

# besides, it turns out I have to massage the files first due to input
model = ['avg_late', 'avg_planck', 'avg_early', 'r1_early', 'r2_early', 'r3_early', 'r4_early', 'r1_planck', 'r2_planck', 'r3_planck', 'r4_planck', 'r1_late', 'r2_late', 'r3_late', 'r4_late']

def do_the_thing(model):
    file_ini = open('../catalinas_codes/cross_21cm_'+str(model)+'.txt','rt')
    file_out = open('./cross_21cm_'+str(model)+'.txt','wt')
    for line in file_ini:
        file_out.write(line.replace(',',' '))
    file_ini.close()
    file_out.close()

    comand = 'python prep_larger_scales.py '+str(model)
    os.system(comand)

for i in range(0, len(model)):
    do_the_thing(model[i])
