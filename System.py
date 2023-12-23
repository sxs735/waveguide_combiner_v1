#%%
import numpy as np
import matplotlib.path as mpath
from Optical_object import *
import open3d as o3d
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
            for num in range(max_iter):
                print(num,len(k_rays))
                index = np.sqrt(np.sum(k_rays[:,1:4]**2,axis = 1))
                direction_cosine = k_rays[:,1:4]/index[:,np.newaxis]
                delta_z = z_layer-k_rays[:,6:7]
                step = delta_z/direction_cosine[:,-1:]
                step = np.min(np.where(step > 0, step, np.inf), axis=1)
                k_rays = k_rays[~np.isinf(step)]
                direction_cosine = direction_cosine[~np.isinf(step)]
                step = step[~np.isinf(step)]
                step = direction_cosine*step[:,np.newaxis]
                start = deepcopy(k_rays[:,4:7]) if save_rays else []
                k_rays[:,4:7] += step
                k_rays[:,4:7] = np.round(k_rays[:,4:7],6)
                end = deepcopy(k_rays[:,4:7]) if save_rays else []
                if save_rays:
                    self.lineset += [np.dstack((start,end))]
                next_krays = []
                for z in z_layer:
                    on_layer_kray = k_rays[np.round(k_rays[:,6],4) == z]
                    k_rays = k_rays[np.round(k_rays[:,6],6) != z]
                    for element in self.layers[z]:
                        mask = element[1].contains_points(on_layer_kray[:,4:6],radius=1E-3)
                        if element[0] and np.sum(mask)>0:
                            next_krays += [element[0].launched(on_layer_kray[mask])]
                        on_layer_kray = on_layer_kray[~mask]
                if next_krays:
                    k_rays = np.vstack(next_krays)
                else:
                    break
                    

    def draw(self):
        if self.lineset:
            lines = np.vstack(self.lineset)
            p1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lines[:,:,0]))
            p2= o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lines[:,:,1]))
            idx = np.vstack((np.arange(len(lines)),np.arange(len(lines)))).T.tolist()#[(i,i) for i in np.arange(len(lines))]
            self.lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(p1,p2,idx)
            self.lineset.paint_uniform_color([1,0,0])

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
        if self.lineset:
            o3d.visualization.draw_geometries([self.lineset] + g_list)
        else:
            o3d.visualization.draw_geometries(g_list)
        self.lineset = []

# %%
