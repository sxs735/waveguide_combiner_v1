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

source = Source([-38,11.53,0.45],-1E-9,[0,0,0,0],[0.525], 
                stokes_vector = [1,0,0,0],
                fov_grid = (1,1),
                spatial_grid = (5,5))

G1 = Grating([[0.3795,11]],[Air,LASF46B], add_order = (1,0), output_order = [[1,'T',1,0],[-1,'R',0,0]],output_option = [1,0,1])
G1._set_simulator('Slanted', 'temp1', [5,4], ['Slant','Height','Duty_a'], grid_size = (1,0.01,0.05), hamonics = (8,0),
                  db_name = 'DB_Slanted1.db')
G1._set_structure([45,0.323,0.25],0)

G2 = Grating([[0.2772,-122.2]],[Air,LASF46B],add_order = (1,0),output_option = [1,0,1])
G2._set_simulator('Binary2', 'temp2', [5,4], ['Height','Duty'], grid_size = (0.01,0.05), hamonics = (8,0),
                  db_name = 'DB_Binary2.db')
G2._set_structure([0.06,0.5],0)

G3 = Grating([[0.3795,104.6]],[Air,LASF46B], add_order = (1,0),output_order = [[-1,'R',0,0],[-1,'T',1,0]],output_option = [1,0,1])
G3._set_simulator('Binary3', 'temp3', [5,4], ['Height','Duty'], grid_size = (0.01,0.05), hamonics = (8,0),
                  db_name = 'DB_Binary3.db')
G3._set_structure([0.06,0.5],0)

F1 = Fresnel_loss([Air,LASF46B],output_option = [1,0])
F2 = Fresnel_loss([LASF46B,Air],output_option = [1,0])
R1 = Receiver()

#%%
system = System(Air)
system.add_source(source)
system.add_element(0,G1,np.array([-38,11.53,0.5]))
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
#system.draw()


# %%
R1.footprint([[-6,6],[-4.5,4.5]])
convolve_image = R1.illuminance([[-6,6,120],[-4.5,4.5,90]],show = True)

vmax = convolve_image.max()
vmin = convolve_image.min()
print(1-(vmax-vmin)/(vmax+vmin))
# %%
