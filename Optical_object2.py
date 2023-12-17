#%%
import numpy as np
from scipy.interpolate import griddata
import subprocess, sqlite3, re, os, time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import matplotlib.path as mpath

def save_dat(file_name,header,order_list,value):
    with open(file_name, 'w') as file:
        file.write(header + '\n')
        for order in order_list:
            if isinstance(order, list):
                file.write(f'{order[0],order[1]}')
            else:
                file.write(f'{order}')
            file.write(' ')
        file.write('\n')
        for val in value:
            file.write(f"{val} ")

def fake_rsoft(command):
    #time.sleep(1)
    pattern = r'prefix=(\S+).*?Hamonics_x=(\S+).*?Hamonics_y=(\S+)'
    matches = re.search(pattern, command)
    if matches:
        prefix, hamonics_x, hamonics_y = matches.groups()
        hamonics_x, hamonics_y = int(float(hamonics_x)), int(float(hamonics_y))
        header = '#'+ command
        order_list = ['none']+np.mgrid[-hamonics_x:hamonics_x+1, -hamonics_y:hamonics_y+1].T.reshape((-1,2)).tolist()
        for suffix in ['_ep_ref_coef.dat','_es_ref_coef.dat','_ep_tra_coef.dat','_es_tra_coef.dat']:
            value = np.zeros((len(order_list)-1)*2+1)
            value[0] = 1
            realimag = np.random.random(size=4)
            A = np.sum(realimag**2)
            A = np.sqrt([np.sum(realimag[:2]**2)/A,np.sum(realimag[2:]**2)/A])
            P = np.arctan2(realimag[[1,3]],realimag[[0,2]])
            AP = 0.5*A*np.exp(1j*P)
            real = np.real(AP)
            imag = np.imag(AP)
            value[21:25] = [real[0],imag[0],real[1],imag[1]]
            save_dat(f'{prefix}{suffix}',header,order_list,value)
    return 

def jones_to_muller(jones):
    u_matrix = 1/np.sqrt(2)*np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,-1j,1j,0]]) 
    muller = [u_matrix @ np.kron(J, np.conjugate(J)) @ np.linalg.inv(u_matrix) for J in jones]
    return np.array(muller)

class Material:
    def __init__(self, name, coefficient):
        self.name = name
        self.coefficient = coefficient

    def sellmeier_equation(self,b1,b2,b3,c1,c2,c3,wavelength):
        square_n = 1+b1*wavelength**2/(wavelength**2-c1)+b2*wavelength**2/(wavelength**2-c2)+b3*wavelength**2/(wavelength**2-c3)
        return np.sqrt(square_n)
    
    def __call__(self,wavelength):
        return self.sellmeier_equation(*self.coefficient,wavelength)

class rays_tool:
    def __init__(self,material = Material('Air',[0,0,0,0,0,0]), 
                 input_format = 'hv',
                 output_format = 'k'):
        '''format: 
                hv: horizontal,vertical
                sp: spherical coord
                k: k_vector
        '''
        self.index = material
        self.input_format = input_format
        self.output_format = output_format

    def convert(self, ray):
        ray = np.asarray(ray)
        index = self.index(ray[:,0])
        if self.input_format =='hv':
            h,v,z = np.deg2rad(ray[:,1:4]).T
            z = np.where(z>=0,1,-1)
            kx = index*np.tan(h)/np.sqrt(1+np.tan(h)**2+np.tan(v)**2)
            ky = index*np.tan(v)/np.sqrt(1+np.tan(h)**2+np.tan(v)**2)
            kz = z*np.sqrt(index**2-kx**2-ky**2)
            
        elif self.input_format == 'sp':
            theta,phi = np.deg2rad(ray[:,1:3]).T
            kx = index*np.sin(theta)*np.cos(phi)
            ky = index*np.sin(theta)*np.sin(phi)
            kz = index*np.cos(theta)
        
        elif self.input_format == 'k':
            index = np.sqrt(np.sum(ray[:,1:4]**2,axis = 1))
            kx,ky,kz = ray[:,1:4].T
            
        else:
            print('format error')
            return None

        ray_output = deepcopy(ray)
        if self.output_format =='hv':
            ray_output[:,1] = np.rad2deg(np.arctan(kx/np.sqrt(index**2-kx**2-ky**2)))
            ray_output[:,2] = np.rad2deg(np.arctan(ky/np.sqrt(index**2-kx**2-ky**2)))
            ray_output[:,3] = np.where(kz>0,1,-1)
            return ray_output

        elif self.output_format =='sp':
            ray_output[:,1] = np.rad2deg(np.arcsin(np.sqrt(kx**2+ky**2)/index))
            ray_output[:,2] = np.rad2deg(np.arctan2(ky,kx))
            ray_output[:,3] = np.where(kz>0,1,-1)
            return ray_output
             
        elif self.output_format =='k':
            ray_output[:,1] = kx
            ray_output[:,2] = ky
            ray_output[:,3] = kz
            return ray_output
        else:
            print('format error')
            return None

class Grating:
    def __init__(self, periods, index, hamonics = (10,0), output_order = ()):
        self.index = index
        add_order = (hamonics[0],0) if np.asarray(periods).shape != (2,2) else hamonics
        self.order = np.mgrid[-add_order[0]:add_order[0]+1, -add_order[1]:add_order[1]+1].reshape((2,-1))
        self.periods = np.asarray(periods)
        self.output_order = output_order

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'periods':
            if self.periods.shape != (2,2):
                self.periods = np.vstack((self.periods.reshape((1,2)),[np.inf,0]))
            g_phi = np.deg2rad(self.periods[:,1])
            self.g_vectors = (1/self.periods[:,0]*np.array([np.cos(g_phi),np.sin(g_phi)])).T
            self.order_gv = self.order.T @ self.g_vectors

    def launched(self, k_in, output_option = 0):
        #k_in: [wavelength,kx,ky,kz,x,y,z,s0,s1,s2,s3]
        k_in,order_gv, mn_order = np.repeat(k_in,len(self.order_gv),axis = 0), np.tile(self.order_gv,(len(k_in),1)),  np.tile(self.order.T,(len(k_in),1))
        z_direction = np.where(k_in[:,3]>0,1,-1)
        n_out = np.where(z_direction==1,self.index[1](k_in[:,0]),self.index[0](k_in[:,0]))
        n_in = np.where(z_direction==1,self.index[0](k_in[:,0]),self.index[1](k_in[:,0]))
        k_out = deepcopy(k_in)
        k_in = np.hstack((k_in,mn_order))
        k_out[:,1:3] += k_out[:,0:1]*order_gv   #k0 + order*wavelength*g_vector
        Tkz2 = n_out**2-(k_out[:,1]**2+k_out[:,2]**2)
        Rkz2 = n_in**2-(k_out[:,1]**2+k_out[:,2]**2)

        # if self.output_order:
        #     T_mode = T_mode if self.output_order[0] == 'T' else np.full(T_mode.shape, -1) 
        #     R_mode = R_mode if self.output_order[0] == 'R' else np.full(R_mode.shape, -1)
        #     specify = np.all(mn_order==self.output_order[1:],axis = 1)
        #     T_mode[~specify] = -1
        #     R_mode[~specify] = -1

        Rk_out, Tk_out = k_out[Rkz2>=0], k_out[Tkz2>0]
        Tk_out[:,3] = (z_direction[Tkz2>0]*np.sqrt(Tkz2[Tkz2>0]))
        Rk_out[:,3] = (-z_direction[Rkz2>=0]*np.sqrt(Rkz2[Rkz2>=0]))
        k_out = np.vstack((Rk_out, Tk_out))

        if k_out.size > 0:
            unique = np.unique(k_out[:,:-4],axis = 0)
            uni_k = []
            for k in unique:
                select = np.all(k_out[:,:-4] == k,axis = 1)
                uni_k.append(np.hstack((k, np.sum(k_out[select,-4:],axis = 0))))
                k_out = k_out[~select]
            k_out = np.asarray(uni_k)
        return k_out

# %%
Air_coefficient = [0,0,0,0,0,0]
LASF46B_coefficient = [2.17988922,0.306495184,1.56882437,0.012580538,0.056719137,105.316538]    #1.9
Air = Material('Air',Air_coefficient)
LASF46B = Material('LASF46B',LASF46B_coefficient)
G1 = Grating([[0.3795,11]],[Air,LASF46B],hamonics = (1,0))

# %%
k_in = np.asarray([[0.525,0,0,1,0,0,0,1,0,0,0],[0.525,0,0,1,0,0,0,1,0,0,0]])
a = G1.launched(k_in)
# %%
