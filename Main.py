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

source = Source([-38,10.53,0.45],-1E-9,[-20,20,-15,15],[0.525], 
                stokes_vector = [1,0,0,0],
                fov_grid = (5,5),
                spatial_grid = (3,3))

G1 = Grating([[0.3795,11]],[Air,LASF46B], add_order = (1,0), output_order = [[1,'T',1,0],[-1,'R',0,0]],output_option = [1,0,1])
G1._set_simulator('Slanted', 'temp1', [5,4], ['Slant','Height','Duty_a'], grid_size = (1,0.01,0.05), harmonics = (8,0),
                  db_name = 'DB_Slanted1.db')
G1._set_structure([45,0.323,0.25])
#G1._set_structure([34.25,0.353,0.25])



class Doe2(Grating):
    def _structure_values(self,position):
        def sigmoid(x):
            return 1 / (1 + np.exp(-4*x))
        values = []
        if hasattr(self,'parameters') and self.windows != 0:
            for i,p in enumerate(self.parameters):
                poly = np.poly1d(p)
                values += [(self.windows[i][1]-self.windows[i][0])*sigmoid(poly(position[:,0]-(-29.2514))) + self.windows[i][0]]
            values = np.array(values)
        else:
            values = np.array([self.parameters]*len(position[:,0])).T
        return values

class Doe3(Grating):
    def _structure_values(self,position):
        def sigmoid(x):
            return 1 / (1 + np.exp(-4*x))
        values = []
        if hasattr(self,'parameters') and self.windows != 0:
            for i,p in enumerate(self.parameters):
                poly = np.poly1d(p)
                values += [(self.windows[i][1]-self.windows[i][0])*sigmoid(poly(position[:,1]-(9.93))) + self.windows[i][0]]
            values = np.array(values)
        else:
            values = np.array([self.parameters]*len(position[:,1])).T
        return values


G2 = Grating([[0.2772,-122.2]],[Air,LASF46B],add_order = (1,0),output_option = [1,0,1])
#G2 = Doe2([[0.2772,-122.2]],[Air,LASF46B],add_order = (1,0),output_option = [1,0,1])
G2._set_simulator('Binary2', 'temp2', [5,4], ['Height','Duty'], grid_size = (0.02,0.05), harmonics = (6,0),
                  db_name = 'DB_Binary2.db')
G3 = Grating([[0.3795,104.6]],[Air,LASF46B], add_order = (1,0),output_order = [[-1,'R',0,0],[-1,'T',1,0]],output_option = [1,0,1])
#G3 = Doe3([[0.3795,104.6]],[Air,LASF46B], add_order = (1,0),output_order = [[-1,'R',0,0],[-1,'T',1,0]],output_option = [1,0,1])
G3._set_simulator('Binary3', 'temp3', [5,4], ['Height','Duty'], grid_size = (0.02,0.05), harmonics = (6,0),
                  db_name = 'DB_Binary3.db')

F1 = Fresnel_loss([Air,LASF46B],output_option = [1,0])
F2 = Fresnel_loss([LASF46B,Air],output_option = [1,0])
R1 = Receiver()

#%%
system = System(Air)
system.add_source(source)
system.add_element(0,G1,np.array([-38,10.53,0.5]))
#system.add_element(0,G2,np.array([[13.24,27.875],[-8.81396,27.875],[-30.2654,17.0954],[-25.8143,10.03],[13.24,10.03],[13.24,27.875]])) #3mm DOE1
system.add_element(0,G2,np.array([[13.24,27.4931],[-5.35688,27.4931],[-29.2514,15.4858],[-25.8143,10.03],[13.24,10.03],[13.24,27.4931]])) #1mm DOE1
system.add_element(0,G3,np.array([[13.24,9.93],[-13.24,9.93],[-13.24,-9.93],[13.24,-9.93],[13.24,9.93]]))
system.add_element(0,F1,np.array([[14.62,28.76],[-40.38,28.76],[-40.38,-11.24],[14.62,-11.24],[14.62,28.76]]))
system.add_element(0.5,F2,np.array([[14.62,28.76],[-40.38,28.76],[-40.38,-11.24],[14.62,-11.24],[14.62,28.76]]))
system.add_element(-20,R1,np.array([[6,4.5],[-6,4.5],[-6,-4.5],[6,-4.5],[6,4.5]]))
#system.add_element(20,None,np.array([[6,4.5],[-6,4.5],[-6,-4.5],[6,-4.5],[6,4.5]]))

# %%
def optfun(var):
    R1.clean()
    G2._set_structure([var[:4],var[4:5]],m_windows=[-29.3,13.3], v_windows=[(0.05,0.4),(0.4,0.6)], modulate = 0)
    G3._set_structure([var[5:9],var[9:10]],m_windows=[9.93,-9.93], v_windows=[(0.05,0.6),(0.4,0.6)], modulate = 1)
    system.run(max_iter = 300,save_rays = False)
    intensity_image = R1.intensity([[-20,20,5],[-15,15,5]])
    vmax = intensity_image.max()
    vmin = intensity_image.min()
    u = (vmax-vmin)/(vmax+vmin)
    e = np.sum(intensity_image)
    merit = u**2+(1-e)**2
    print('u=%.3f,e=%.6f,m=%.6f'%(1-u,e,merit))
    return merit
#%%
t0 = time.time()
optfun([0.82015544,-0.17507922,0.149626,-0.74469135,0.46787374,0.5174115,0.50533741,-0.06746517,-0.98599707,-0.91048808])
print(time.time()-t0)
#%%
from scipy.optimize import differential_evolution
bounds = [(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),
          (-1,1),(-1,1),(-1,1),(-1,1),(-1,1)]
#result = minimize(optfun, [-30,0.1,0.3],bounds = bounds,method='Nelder-Mead')
result = differential_evolution(optfun, bounds = bounds)
print(result.x)

#%%
t0 = time.time()
optfun(result.x)
print(time.time()-t0)
#%%
system.draw()


# %%
#R1.footprint([[-6,6],[-4.5,4.5]])
convolve_image = R1.illuminance([[-6,6,120],[-4.5,4.5,90]],show = True)
intensity_image = R1.intensity([[-20,20,5],[-15,15,5]],show = True)
vmax = convolve_image.max()
vmin = convolve_image.min()
print(1-(vmax-vmin)/(vmax+vmin),np.sum(convolve_image))
vmax = intensity_image.max()
vmin = intensity_image.min()
print(1-(vmax-vmin)/(vmax+vmin),np.sum(intensity_image))
# %%
