#%%
from Optical_object import *
from System import *
import time

Air_coefficient = [0,0,0,0,0,0]
NC17_coefficient = [1.85307378, 0.000139589850,1666.40582,0.0145009144,0.0613202273,47378.6097]
Air = Material('Air',Air_coefficient)
NC17 = Material('LASF46B',NC17_coefficient)



source = Source([[0.00,4.50],[4.00,4.50],[4.00,-4.50],[0.00,-4.50],[0.00,4.50]],-1E-3,
                [-2,22,-9,9],[0.525], stokes_vector = [1,0,0,0],
                fov_grid = (5,5),spatial_grid = (5,5))
# source = Source([3,-1,0],-1E-3,
#                 [1,1,0,0],[0.525], stokes_vector = [1,0,0,0],
#                 fov_grid = (1,1),spatial_grid = (1,1), shrink = 0)

G1_G = Grating([[0.455,0]],[Air,NC17], add_order = (1,0))
G2A1_G = Grating([[0.380,-120]],[Air,NC17],add_order = (1,0))
G2B1_G = Grating([[0.380, 120]],[Air,NC17],add_order = (1,0))
G2A2_G = Grating([[0.380,-120]],[Air,NC17],add_order = (1,0))
G2B2_G = Grating([[0.380, 120]],[Air,NC17],add_order = (1,0))
G3_G = Grating([[0.455,180]],[Air,NC17], add_order = (1,0))

G1_R = Grating([[0.565,0]],[Air,NC17], add_order = (1,0))
G2A1_R = Grating([[0.485,-120]],[Air,NC17],add_order = (1,0))
G2B1_R = Grating([[0.485, 120]],[Air,NC17],add_order = (1,0))
G2A2_R = Grating([[0.485,-120]],[Air,NC17],add_order = (1,0))
G2B2_R = Grating([[0.485, 120]],[Air,NC17],add_order = (1,0))
G3_R = Grating([[0.565,180]],[Air,NC17], add_order = (1,0))

F1 = Fresnel_loss([Air,NC17])
F2 = Fresnel_loss([NC17,Air])
R1 = Receiver()

system = System(Air)
system.add_source(source)
system.add_element(0,G1_G,np.array([[0.00,4.50],[4.00,4.50],[4.00,-4.50],[0.00,-4.50],[0.00,4.50]]))
system.add_element(0,G2A1_G,np.array([[9.00,6.30],[15.00,7.17],[15.00,0.50],[10.95,0.50],[9.00,6.30]]))
system.add_element(0,G2B1_G,np.array([[9.00,-6.30],[15.00,-7.17],[15.00,-0.50],[10.95,-0.50],[9.00,-6.30]]))
system.add_element(0,G2A2_G,np.array([[19.00,-0.50],[15.50,-9.00],[21.50,-9.00],[21.50,-0.50],[19.00,-0.50]]))
system.add_element(0,G2B2_G,np.array([[19.00,0.50],[15.50,9.00],[21.50,9.00],[21.50,0.50],[19.00,0.50]]))
system.add_element(0,G3_G,np.array([[22.50,7.00],[43.50,7.00],[43.50,-7.00],[22.50,-7.00],[22.50,7.00]]))
system.add_element(0,F1,np.array([[0,5.5],[16,10],[45,10],[45,-10],[16,-10],[0,-5.5],[0,5.5]]))
system.add_element(1.375,F2,np.array([[0,5.5],[16,10],[45,10],[45,-10],[16,-10],[0,-5.5],[0,5.5]]))

#system.add_element(1.475,G1_R,np.array([[0.00,4.50],[4.00,4.50],[4.00,-4.50],[0.00,-4.50],[0.00,4.50]]))
# system.add_element(1.475,G2A1_R,np.array([[9.00,6.30],[15.00,7.17],[15.00,0.50],[10.95,0.50],[9.00,6.30]]))
# system.add_element(1.475,G2B1_R,np.array([[9.00,-6.30],[15.00,-7.17],[15.00,-0.50],[10.95,-0.50],[9.00,-6.30]]))
# system.add_element(1.475,G2A2_R,np.array([[19.00,-0.50],[15.50,-9.00],[21.50,-9.00],[21.50,-0.50],[19.00,-0.50]]))
# system.add_element(1.475,G2B2_R,np.array([[19.00,0.50],[15.50,9.00],[21.50,9.00],[21.50,0.50],[19.00,0.50]]))
# system.add_element(1.475,G3_R,np.array([[22.50,7.00],[43.50,7.00],[43.50,-7.00],[22.50,-7.00],[22.50,7.00]]))
# system.add_element(1.475,F1,np.array([[0,5.5],[16,10],[45,10],[45,-10],[16,-10],[0,-5.5],[0,5.5]]))
# system.add_element(2.85,F2,np.array([[0,5.5],[16,10],[45,10],[45,-10],[16,-10],[0,-5.5],[0,5.5]]))

system.add_element(20,R1,np.array([[22.50,7.00],[43.50,7.00],[43.50,-7.00],[22.50,-7.00],[22.50,7.00]]))

#system.draw()


# %%
t0 = time.time()
system.run(max_iter = 50,save_rays = True)
print(time.time()-t0)
# %%
system.draw()
# %%
