#%%
from Optical_object import *
from System import *
import time

Air_coefficient = [0,0,0,0,0,0]
LASF46B_coefficient = [2.17988922,0.306495184,1.56882437,0.012580538,0.056719137,105.316538]    #1.9
SF69_coefficient = [1.62594647,0.235927609,1.674346230,0.01216966770,0.0600710405,145.6519080]  #1.7
Nc_coefficient = [1.9**2-1,0,0,0,0,0]
Air = Material('Air',Air_coefficient)
LASF46B = Material('LASF46B',LASF46B_coefficient)
NC = Material('nc',Nc_coefficient)

source = Source([-38,11.53,1],-0.1,[-20,20,-15,15],[0.525], 
                stokes_vector = [1,0,0,0],
                fov_grid = (5,5),
                spatial_grid = (7,7))

G1 = Grating([[0.3795,11]],[Air,LASF46B], add_order = (1,0))
G2 = Grating([[0.2772,-122.2]],[Air,LASF46B],add_order = (1,0))
G3 = Grating([[0.3795,104.6]],[Air,LASF46B], add_order = (1,0))
F1 = Fresnel_loss([Air,LASF46B])
F2 = Fresnel_loss([LASF46B,Air])
R1 = Receiver()

#%%
system = System(Air)
system.add_source(source)
system.add_element(0,G1,np.array([-38,11.53,1.5]))
system.add_element(0,G2,np.array([[13.24,27.875],[-8.81396,27.875],[-30.2654,17.0954],[-25.8143,10.03],[13.24,10.03],[13.24,27.875]]))
system.add_element(0,G3,np.array([[13.24,9.93],[-13.24,9.93],[-13.24,-9.93],[13.24,-9.93],[13.24,9.93]]))
system.add_element(0,F1,np.array([[14.62,28.76],[-40.38,28.76],[-40.38,-11.24],[14.62,-11.24],[14.62,28.76]]))
system.add_element(0.5,F2,np.array([[14.62,28.76],[-40.38,28.76],[-40.38,-11.24],[14.62,-11.24],[14.62,28.76]]))
system.add_element(-20,R1,np.array([[6,4.5],[-6,4.5],[-6,-4.5],[6,-4.5],[6,4.5]]))
#system.add_element(20,None,np.array([[6,4.5],[-6,4.5],[-6,-4.5],[6,-4.5],[6,4.5]]))

# %%
t0 = time.time()
system.run(max_iter = 300,save_rays = False)
print(time.time()-t0)

#%%
system.draw()

# %%
import matplotlib.pyplot as plt
eyebox = np.vstack(R1.store)

plt.scatter(eyebox[:,4] , eyebox[:,5])

plt.show()