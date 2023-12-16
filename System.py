#%%
import numpy as np
import matplotlib.path as mpath
from Optical_object import *
import open3d as o3d
import open3d.visualization.rendering as rendering
import time

class System:
    def __init__(self, material, boundary = [[-50,20],[-20,50],[-30,20]]):
        self.index = material
        self.boundary = np.array(boundary)
        self.layers = {}
        self.lineset = []

    def add_source(self,source):
        self.source = source

    def add_element(self,z,object,shape):
        polygon = mpath.Path.circle(shape[:2], shape[2]) if shape.ndim == 1 else mpath.Path(shape)
        if z in self.layers:
            self.layers[z] += [[object,polygon]]
        else:
            self.layers.update({z : [[object,polygon]]})

    def run(self,max_iter = 2, save_rays = False):
        if hasattr(self,'source'):
            k_rays = self.source.launch()
            z_layer = np.asarray(list(self.layers.keys()))
            
            for k_rays_i in k_rays:
                t0 = time.time()
                print(k_rays_i[1:4])
                k_rays_i = k_rays_i[np.newaxis,:]
                for num in range(max_iter):
                    index = np.sqrt(np.sum(k_rays_i[:,1:4]**2,axis = 1))
                    direction_cosine = k_rays_i[:,1:4]/index[:,np.newaxis]
                    delta_z = z_layer-k_rays_i[:,6:7]
                    step = delta_z/direction_cosine[:,-1:]
                    step = np.min(np.where(step > 0, step, np.inf), axis=1)
                    k_rays_i = k_rays_i[~np.isinf(step)]
                    direction_cosine = direction_cosine[~np.isinf(step)]
                    step = step[~np.isinf(step)]
                    step = direction_cosine*step[:,np.newaxis]
                    start = deepcopy(k_rays_i[:,4:7]) if save_rays else []
                    k_rays_i[:,4:7] += step
                    k_rays_i[:,4:7] = np.round(k_rays_i[:,4:7],6)
                    end = deepcopy(k_rays_i[:,4:7]) if save_rays else []
                    if save_rays:
                        for i in range(len(k_rays_i)):
                            ray = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector([start[i],end[i]]),
                                                    lines=o3d.utility.Vector2iVector([[0,1]]))
                            ray.paint_uniform_color([1,0,0])
                            self.lineset += [ray]
                    next_krays = []
                    for z in z_layer:
                        on_layer_kray = k_rays_i[np.round(k_rays_i[:,6],4) == z]
                        k_rays_i = k_rays_i[np.round(k_rays_i[:,6],6) != z]
                        for element in self.layers[z]:
                            mask = element[1].contains_points(on_layer_kray[:,4:6],radius=1E-3)
                            if element[0]:
                                next_krays += [element[0].launched(on_layer_kray[mask])]
                            on_layer_kray = on_layer_kray[~mask]
                    k_rays_i = np.vstack(next_krays)
                    if len(k_rays_i)==0:
                        break
                print(time.time()-t0)
                    

    def draw(self):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=self.boundary.min(axis = 1))
        g_list = [mesh_frame]
        if hasattr(self,'source'):
            lines = [[i,i+1] for i in range(len(self.source.polygon.vertices) - 1)]
            points = np.column_stack((self.source.polygon.vertices, self.source.z*np.ones(len(self.source.polygon.vertices))))
            g_list += [o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                              lines=o3d.utility.Vector2iVector(lines))]

        for z in self.layers.keys():
            for shape in self.layers[z]:
                lines = [[i,i+1] for i in range(len(shape[1].vertices) - 2)]+[[len(shape[1].vertices)-2,0]]
                points = np.column_stack((shape[1].vertices, z*np.ones(len(shape[1]))))[:-1]

                g_list += [o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                                  lines=o3d.utility.Vector2iVector(lines))]
        x,y,z = np.meshgrid(np.linspace(*self.boundary[0],2),
                            np.linspace(*self.boundary[1],2),
                            np.linspace(*self.boundary[2],2))
        boundary = np.vstack((x.reshape(-1),y.reshape(-1),z.reshape(-1))).reshape((-1,3))
        boundary = o3d.cpu.pybind.geometry.PointCloud(o3d.utility.Vector3dVector(boundary)).get_axis_aligned_bounding_box()
        g_list += [o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(boundary)]
        o3d.visualization.draw_geometries(self.lineset + g_list)


#%%
Air_coefficient = [0,0,0,0,0,0]
LASF46B_coefficient = [2.17988922,0.306495184,1.56882437,0.012580538,0.056719137,105.316538]    #1.9
SF69_coefficient = [1.62594647,0.235927609,1.674346230,0.01216966770,0.0600710405,145.6519080]  #1.7
Nc_coefficient = [1.9**2-1,0,0,0,0,0]
Air = Material('Air',Air_coefficient)
LASF46B = Material('LASF46B',LASF46B_coefficient)
NC = Material('nc',Nc_coefficient)

source = Source([-38,11.53,1.5],-0.1,[-20,20,-15,15],[0.525], 
                stokes_vector = [1,0,0,0],
                fov_grid = (5,5),
                spatial_grid = (5,5))

G1 = Grating([[0.3795,11]],[Air,LASF46B],delta_order = (1,0))
G2 = Grating([[0.2772,-122.2]],[Air,LASF46B],delta_order = (1,0))
G3 = Grating([[0.3795,104.6]],[Air,LASF46B],delta_order = (1,0))
F1 = Fresnel_loss([Air,LASF46B])
F2 = Fresnel_loss([LASF46B,Air])

#%%
system = System(Air)
system.add_source(source)
system.add_element(0,G1,np.array([-38,11.53,1.5]))
system.add_element(0,G2,np.array([[13.24,27.875],[-8.81396,27.875],[-30.2654,17.0954],[-25.8143,10.03],[13.24,10.03],[13.24,27.875]]))
system.add_element(0,G3,np.array([[13.24,9.93],[-13.24,9.93],[-13.24,-9.93],[13.24,-9.93],[13.24,9.93]]))
system.add_element(0,F1,np.array([[14.62,28.76],[-40.38,28.76],[-40.38,-11.24],[14.62,-11.24],[14.62,28.76]]))
system.add_element(0.5,F2,np.array([[14.62,28.76],[-40.38,28.76],[-40.38,-11.24],[14.62,-11.24],[14.62,28.76]]))
system.add_element(-20,None,np.array([[6,4.5],[-6,4.5],[-6,-4.5],[6,-4.5],[6,4.5]]))
system.add_element(20,None,np.array([[6,4.5],[-6,4.5],[-6,-4.5],[6,-4.5],[6,4.5]]))
# %%
system.draw()
# %%
t0 = time.time()
system.run(max_iter = 300,save_rays = False)
print(time.time()-t0)

# %%
kin = source.launch()[2:3]
k1 = G1.launched(kin)
F2.launched(np.array(k1))

# %%
for i in system.lineset:
    print(np.asarray(i.points))
# %%
