import numpy as np

xa = np.linspace(0.,90.,91)
for i in range(18):
    g = open('ff_ray_'+str(i)+'.dat','w')
    for x in xa:
        table = np.loadtxt('flux_obs_ang_'+str(x)+'_nrays_18.dat',unpack=True,usecols=([4])) 
        g.write('%20s %20s\n'%(x,table[i]))
    g.close()
