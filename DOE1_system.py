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

source = Source([0,0,0.1],-0.1,[-20,20,-15,15],[0.525], 
                stokes_vector = [1,0,0,0],
                fov_grid = (3,3),
                spatial_grid = (3,3))

# source = Source([0,0,0],-0.1,[0,0,0,0],[0.525], 
#                 stokes_vector = [1,0,0,0],
#                 fov_grid = (1,1),
#                 spatial_grid = (1,1))

G1 = Grating([[0.3795,11]],[Air,LASF46B], add_order = (1,0), output_order = [[1,'T',1,0],[-1,'R',0,0]],output_option = [1,0,0])
G1._set_simulator('Slanted', 'temp', [5,4], ['Slant','Height','Duty_a'], grid_size = (1,0.01,0.05), db_name = 'DB_Slant_3795_LASF46B_19.db' , hamonics = (8,0))
E3 = Extracter([[0.3795,191]],[Air,LASF46B], add_order = (1,0), output_order = [[-1,'T',1,0]])
F1 = Fresnel_loss([Air,LASF46B])
F2 = Fresnel_loss([LASF46B,Air])
R1 = Receiver()

system = System(Air)
system.add_source(source)
system.add_element(0,G1,np.array([0,0,0.15]))
system.add_element(0,E3,np.array([[10,5],[2,5],[2,-5],[10,-5],[10,5]]))
system.add_element(0,F1,np.array([[10,5],[-2,5],[-2,-5],[10,-5],[10,5]]))
system.add_element(0.5,F2,np.array([[10,5],[-2,5],[-2,-5],[10,-5],[10,5]]))
system.add_element(-0.1,R1,np.array([[10,5],[1.5,5],[1.5,-5],[10,-5],[10,5]]))

#%%
from scipy.optimize import minimize

def optfun(var):
    G1._set_structure(var,0)
    system.run(max_iter = 30,save_rays = True)
    k_out = R1()
    R1.clean()
    unique, idx, counts = np.unique(k_out[:,:4], axis = 0, return_index=True, return_counts = True)
    overlap_rays = unique[counts>1]
    if overlap_rays.size > 0:
        k_unique = k_out[idx[counts==1]] 
        if k_unique.size == 0:
            k_unique = np.empty((0, 8))
        stoke_vector = [np.sum(k_out[np.all(k_out[:,:4] == k,axis = 1),-4:],axis = 0) for k in overlap_rays]
        k_out = np.vstack((k_unique,np.hstack((overlap_rays, stoke_vector))))
    res = k_out[:,-4]
    energy = np.sum(res)
    uniform = (res.max()-res.min())/(res.max()+res.min())
    print('e=%.4f,u=%.4f'%(energy,uniform))
    return (1-energy)**2+uniform**2

result = minimize(optfun, [40,0.37,0.2],bounds = [(20,50),(0.2,0.6),(0.2,0.5)],
                  method='Nelder-Mead')
res = optfun(result.x)
print(res)

#%%
system.draw()

# %%
ktohv = rays_tool(material = Material('Air',[0,0,0,0,0,0]),
                  input_format = 'k',output_format = 'hv')

b = ktohv.convert(a)
# %%
