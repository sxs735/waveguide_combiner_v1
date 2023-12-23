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

class Source:
    def __init__(self,shape,z,fov_box,wavelength_list,
                 direction = 1, stokes_vector = [1,0,0,0],
                 material = Material('Air',[0,0,0,0,0,0]),
                 fov_grid = (5,5),
                 spatial_grid = (3,3),shrink = 1E-3):
        shape = np.array(shape)
        self.z = z
        self.fov_box = np.asarray(fov_box)
        self.wavelength_list = np.asarray(wavelength_list)
        self.fgrid = np.asarray(fov_grid)
        self.sgrid = np.asarray(spatial_grid)
        self.material = material
        self.stokes_vector = [stokes_vector]
        self.shrink = 0 if shape.ndim == 1 else shrink 

        #points
        self.polygon = mpath.Path.circle(shape[:2], shape[2]) if shape.ndim == 1 else mpath.Path(shape)
        self.range = np.vstack((self.polygon.vertices.min(axis = 0)+self.shrink,self.polygon.vertices.max(axis = 0)-self.shrink)).T
        x,y = np.meshgrid(np.linspace(*self.range[0],self.sgrid[0]),np.linspace(*self.range[1],self.sgrid[1]))
        self.points = np.vstack((x.reshape(-1),y.reshape(-1),z*np.ones_like(x.reshape(-1)))).T
        if len(self.points) == 1 and shape.ndim == 1 and shape[2] == 0:
            pass
        else:
            mask = self.polygon.contains_points(self.points[:,:2], radius=1E-3)
            self.points = self.points[mask]
        #rays
        h,v,w = np.meshgrid(np.linspace(*self.fov_box[:2],self.fgrid[0]),
                            np.linspace(*self.fov_box[2:],self.fgrid[1]),
                            self.wavelength_list)
        self.rays = np.vstack((w.reshape(-1),h.reshape(-1),v.reshape(-1), direction*np.ones_like(h.reshape(-1)))).T

    def launch(self):
        k_rays = rays_tool(self.material)
        k_rays = k_rays.convert(self.rays)
        rays = np.hstack((k_rays.repeat(self.points.shape[0],axis = 0),np.tile(self.points,(k_rays.shape[0],1))))
        rays = np.hstack((rays,np.repeat(self.stokes_vector/rays.shape[0],rays.shape[0],axis = 0)))
        return rays

class Grating:
    class Simulator:

        @staticmethod
        def rs_cmd(indfile, prefix, variable = {}, hide = True):
            '''output_setting is a dictionary
            #launch
                free_space_wavelength
                rcwa_primary_direction=, 0(+x), 1(-x), 2(+y), 3(-y), 4(+z), 5(-z)
                launch_angle
                launch_theta
                rcwa_launch_pol
                rcwa_launch_delta_phase
            #output_setting
                rcwa_output_e_coef_format=1,2,3(RI),4(AP)
                rcwa_output_option=,1(single value), 2(vs wavelength),3(vs phi),4(vs theta)
                rcwa_output_total_refl=,0(enable),1(disable)
                rcwa_output_total_trans=,0(enable),1(disable)
                rcwa_output_absorption=,0(enable),1(disable)
                rcwa_output_diff_refl=,0(enable),1(disable)
                rcwa_output_diff_trans=,0(enable),1(disable)
                rcwa_ref_order_x
                rcwa_ref_order_y
                rcwa_tra_order_x
                rcwa_tra_order_y
                rcwa_variation_max
                rcwa_variation_min
                rcwa_variation_step
            '''
            hide = ' -hide ' if hide else ' '
            basic_cmd = f"dfmod{hide}{indfile}.ind prefix={prefix} wait=0"
            variable_cmd = ''
            for var in variable.keys():
                variable_cmd += ' '+var+'='+'%.4f'%(variable[var])
            return basic_cmd + variable_cmd
            
        @staticmethod
        def count_decimal(number):
            number_str = str(number)
            return len(number_str.split('.')[1]) if '.' in number_str else 0
        
        def __init__(self, indfile, prefix, z_direction, symbols, hamonics = (10,0)):
            self.indfile = indfile 
            self.prefix = prefix
            self.z_direction = z_direction
            self.symbols_base = ['rcwa_primary_direction','free_space_wavelength','launch_angle','launch_theta'],['INTEGER','REAL','REAL','REAL']
            self.symbols_values = ['order_m','order_n','r_matrix','t_matrix'],['INTEGER','INTEGER','BLOB','BLOB']
            self.symbols = symbols,['REAL']*len(symbols)
            self.hamonics = hamonics

        def _generate_cmd(self,variable_list):
            symbols = self.symbols_base[0]+self.symbols[0]
            pol = {'p':0,'s':90}
            cmd = []
            for p in ['p','s']:
                for i,var in enumerate(variable_list):
                    dict_ = dict(zip(symbols, var[:len(symbols)]))
                    dict_['rcwa_launch_pol'] = pol[p]
                    dict_['Hamonics_x'] = self.hamonics[0]
                    dict_['Hamonics_y'] = self.hamonics[1]
                    cmd += [self.rs_cmd(self.indfile,self.prefix+f"_{i}_{p}", variable = dict_)]
            return cmd
        
        def _set_db(self,db_name,table,grid_size):
            self.table = table
            self.grid_size = np.asarray(grid_size)
            self.decimal = np.max([self.count_decimal(number) for number in grid_size])
            set_column = ','.join([f'{i} {j} NOT NULL' for i,j in zip(self.symbols_base[0]+self.symbols[0]+self.symbols_values[0],
                                                                    self.symbols_base[1]+self.symbols[1]+self.symbols_values[1])])
            self.db = sqlite3.connect(db_name)
            self.cursor = self.db.cursor()
            self.cursor.execute(f'CREATE TABLE IF NOT EXISTS {self.table} (id INTEGER PRIMARY KEY, {set_column})')
            
            #{(rcwa_primary_direction,free_space_wavelength,launch_angle,launch_theta,*symbols): {(order_m,order_n): [r_matrix,t_matrix]}}
            self.dict_db = {}
            self.cursor.execute(f'SELECT * FROM {self.table}')
            database = self.cursor.fetchall()
            for row in database:
                if row[1:-4] in self.dict_db:
                    self.dict_db[row[1:-4]].update({row[-4:-2]: row[-2:]})
                else:
                    self.dict_db[row[1:-4]] = {row[-4:-2]: list(row[-2:])}

        def _search_db(self,key):
            try:
                result = list(self.dict_db[tuple(key[:-2])][tuple(key[-2:])])
                result[-2] = np.frombuffer(result[-2], dtype=np.complex128).reshape((2, 2))
                result[-1] = np.frombuffer(result[-1], dtype=np.complex128).reshape((2, 2))
            except KeyError:
                result = [np.full((2,2),np.nan)]*2
            return result
        
        def _to_db(self,values):
            items_str = self.symbols_base[0] + self.symbols[0] + self.symbols_values[0]
            placeholders = ','.join('?'*len(items_str))
            items_str = ','.join(items_str)
            self.cursor.executemany(f'INSERT INTO {self.table} ({items_str}) VALUES ({placeholders})', values)
            self.db.commit()
            for row in values:
                row = tuple(row)
                if row[:-4] in self.dict_db:
                    self.dict_db[row[:-4]].update({row[-4:-2]: row[-2:]})
                else:
                    self.dict_db[row[:-4]] = {row[-4:-2]: row[-2:]}

        def _delete(self):
            file_list = os.listdir()
            for file_name in file_list:
                if self.prefix in file_name:
                    os.remove(file_name)

        def _compute(self,variable_list, save_to_db = False):
            commands = self._generate_cmd(variable_list)
            symbols = self.symbols_base[0]+self.symbols[0]
            #for cmd in commands:
            #     fake_rsoft(cmd)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                executor.map(fake_rsoft, commands)
                #executor.map(subprocess.call, commands)

            output = []
            order = np.hstack(np.mgrid[-self.hamonics[0]:self.hamonics[0]+1, -self.hamonics[1]:self.hamonics[1]+1])
            for i,var in enumerate(variable_list):
                var = var[:len(symbols)]
                p_ep_r = np.loadtxt(self.prefix+f"_{i}_p_ep_ref_coef.dat",skiprows=2)[1:].reshape((-1,2))
                p_es_r = np.loadtxt(self.prefix+f"_{i}_p_es_ref_coef.dat",skiprows=2)[1:].reshape((-1,2))
                s_ep_r = np.loadtxt(self.prefix+f"_{i}_s_ep_ref_coef.dat",skiprows=2)[1:].reshape((-1,2))
                s_es_r = np.loadtxt(self.prefix+f"_{i}_s_es_ref_coef.dat",skiprows=2)[1:].reshape((-1,2))
                p_ep_t = np.loadtxt(self.prefix+f"_{i}_p_ep_tra_coef.dat",skiprows=2)[1:].reshape((-1,2))
                p_es_t = np.loadtxt(self.prefix+f"_{i}_p_es_tra_coef.dat",skiprows=2)[1:].reshape((-1,2))
                s_ep_t = np.loadtxt(self.prefix+f"_{i}_s_ep_tra_coef.dat",skiprows=2)[1:].reshape((-1,2))
                s_es_t = np.loadtxt(self.prefix+f"_{i}_s_es_tra_coef.dat",skiprows=2)[1:].reshape((-1,2))
                p_ep_r = p_ep_r[:,0]+p_ep_r[:,1]*1j
                p_es_r = p_es_r[:,0]+p_es_r[:,1]*1j
                s_ep_r = s_ep_r[:,0]+s_ep_r[:,1]*1j
                s_es_r = s_es_r[:,0]+s_es_r[:,1]*1j
                p_ep_t = p_ep_t[:,0]+p_ep_t[:,1]*1j
                p_es_t = p_es_t[:,0]+p_es_t[:,1]*1j
                s_ep_t = s_ep_t[:,0]+s_ep_t[:,1]*1j
                s_es_t = s_es_t[:,0]+s_es_t[:,1]*1j
                mask = ((np.abs(p_ep_r)>0)+(np.abs(p_es_r)>0)+(np.abs(s_ep_r)>0)+(np.abs(s_es_r)>0)
                        +(np.abs(p_ep_t)>0)+(np.abs(p_es_t)>0)+(np.abs(s_ep_t)>0)+(np.abs(s_es_t)>0))>0
                p_ep_r = p_ep_r[mask]
                p_es_r = p_es_r[mask]
                s_ep_r = s_ep_r[mask]
                s_es_r = s_es_r[mask]
                p_ep_t = p_ep_t[mask]
                p_es_t = p_es_t[mask]
                s_ep_t = s_ep_t[mask]
                s_es_t = s_es_t[mask]

                for j,mn in enumerate(order[mask]):
                    r_matrix = sqlite3.Binary(np.array([[p_ep_r[j],s_ep_r[j]],[p_es_r[j],s_es_r[j]]]).tobytes())
                    t_matrix = sqlite3.Binary(np.array([[p_ep_t[j],s_ep_t[j]],[p_es_t[j],s_es_t[j]]]).tobytes())
                    output += [[*var,int(mn[0]), int(mn[1]),r_matrix,t_matrix]]
            
            self._delete()
            
            if save_to_db:
                self._to_db(output)
            else:
                output = np.asarray(output,dtype = object)
                variable_list = np.asarray(variable_list)
                indices = np.any(np.all(output[:, :-2][:, None, :].astype(float) == variable_list, axis=-1),axis = 1)
                output = [[np.frombuffer(out[-2], dtype=np.complex128).reshape((2, 2)),
                        np.frombuffer(out[-1], dtype=np.complex128).reshape((2, 2))] for out in output[indices]]
                return np.asarray(output)
            
        def _estimate(self, variables):
            #variables = [[direction,wavelength,theta, phi,*symbol,order_m,order_n]]
            variables = np.asarray(variables)
            if hasattr(self,'db'):
                #buliding a boundary box
                floor = np.round(variables[:,5:5+len(self.grid_size)]//self.grid_size*self.grid_size,self.decimal)
                ceil = np.round(floor + self.grid_size ,self.decimal)
                boxes = np.dstack((floor,ceil))
                #search for the var that needs computation.
                var_list = np.vstack([np.vstack(np.meshgrid(*box)).reshape((len(box), -1)).T for box in boxes])
                repeat = len(var_list)//len(variables)
                var_list = np.hstack([variables[:,:4].repeat(repeat, axis=0), var_list, variables[:,-2:].repeat(repeat, axis=0)])
                computed_list = np.unique(var_list, axis = 0)
                res_list = np.asarray([self._search_db(var_i) for var_i in computed_list])
                #compute
                if np.any(np.isnan(res_list)):
                    not_exist = np.any(np.isnan(res_list),axis = (1,2,3))
                    self._compute(computed_list[:,:-2][not_exist],save_to_db = True)
                #get results
                res_list = np.asarray([self._search_db(var_i) for var_i in var_list])
                if np.any(np.isnan(res_list)):
                    not_exist = np.any(np.isnan(res_list),axis = (1,2,3))
                    res_list[not_exist] = np.zeros(res_list[not_exist].shape)
                #interpolation
                var_list = var_list.reshape((len(variables),-1,var_list.shape[-1]))[:,:,4:4+len(self.symbols[0])]
                res_list = res_list.reshape((len(variables),-1,*res_list.shape[1:]))
                grid_res = []
                for i,var in enumerate(variables):
                    amp = np.abs(res_list[i])
                    phase = np.angle(res_list[i])
                    grid_amp = griddata(var_list[i], amp, var[5:5+len(self.symbols[0])],method='linear')
                    grid_phase = griddata(var_list[i], phase, var[5:5+len(self.symbols[0])],method='linear')
                    grid_res += [grid_amp*np.exp(1j*grid_phase)]
                return np.asarray(grid_res).reshape((-1,2,2,2))
            else:
                if variables.ndim <=1:
                    variables = variables[np.newaxis,:]
                return self._compute(variables)
            
        def _close_db(self):
            if hasattr(self,'db'):
                self.db.close()
                del self.db


    def __init__(self, periods, index, add_order = (1,0)):
        self.index = index
        add_order = (add_order[0],0) if np.asarray(periods).shape != (2,2) else add_order
        self.order = np.mgrid[-add_order[0]:add_order[0]+1, -add_order[1]:add_order[1]+1].reshape((2,-1))
        self.periods = np.asarray(periods)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'periods':
            if self.periods.shape != (2,2):
                self.periods = np.vstack((self.periods.reshape((1,2)),[np.inf,0]))
            g_phi = np.deg2rad(self.periods[:,1])
            self.g_vectors = (1/self.periods[:,0]*np.array([np.cos(g_phi),np.sin(g_phi)])).T
            self.order_gv = self.order.T @ self.g_vectors

    def _set_simulator(self, indfile, prefix, z_direction, symbols, grid_size = (), db_name = '' , hamonics = (10,0)):
        self.simulator = self.Simulator(indfile, prefix, z_direction, symbols, hamonics = hamonics)
        if db_name and grid_size:
            self.simulator._set_db(db_name,'rayleigh_matrices', grid_size)

    def _set_structure(self,parameters,windows):
        self.parameters = parameters
        self.windows = windows

    def _structure_values(self,position):
        def sigmoid(x):
            return 1 / (1 + np.exp(-4*x))
        values = []
        if hasattr(self,'parameters') and self.windows != 0:
            for i,p in enumerate(self.parameters):
                poly = np.poly1d(p)
                values += [(self.windows[i][1]-self.windows[i][0])*sigmoid(poly(position[:,0])) + self.windows[i][0]]
            values = np.array(values)
        else:
            values = np.array([self.parameters]*len(position[:,0])).T
        return values

    def _k_to_rsoft(self,k_in):
        #[direction, wavelength, theta, phi,*symbol, order_m, order_n]
        direction = np.where(k_in[:,3]>=0,*self.simulator.z_direction)
        ray_sp = rays_tool(input_format = 'k',output_format = 'sp')
        k_in_sp = ray_sp.convert(k_in)
        k_in_sp[:,2] -= self.periods[0,1]
        symbol_values =  self._structure_values(k_in_sp[:,4:6])
        return np.vstack((direction,*np.round(k_in_sp[:,:4].T,4),*symbol_values,*k_in_sp[:,-2:].T)).T

    def launched(self, k_in, output_order = (), output_option = 0):
        #k_in: [wavelength,kx,ky,kz,x,y,z,s0,s1,s2,s3]
        k_in,order_gv, mn_order = np.repeat(k_in,len(self.order_gv),axis = 0), np.tile(self.order_gv,(len(k_in),1)),  np.tile(self.order.T,(len(k_in),1))
        z_direction = np.where(k_in[:,3]>0,1,-1)
        n_out = np.where(z_direction==1,self.index[1](k_in[:,0]),self.index[0](k_in[:,0]))
        n_in = np.where(z_direction==1,self.index[0](k_in[:,0]),self.index[1](k_in[:,0]))
        k_out = deepcopy(k_in)
        k_in = np.hstack((k_in,mn_order))
        k_out[:,1:3] += k_out[:,0:1]*order_gv   #k0 + order*wavelength*g_vector
        Tkz2 = n_out**2-(k_out[:,1]**2+k_out[:,2]**2)   #Transmission kz**2
        Rkz2 = n_in**2-(k_out[:,1]**2+k_out[:,2]**2)    #Reflection kz**2

        if output_order:
            Tkz2 = Tkz2 if output_order[0] == 'T' else np.full(Tkz2.shape, -1) 
            Rkz2 = Rkz2 if output_order[0] == 'R' else np.full(Rkz2.shape, -1)
            specify = np.all(mn_order==output_order[1:],axis = 1)
            Tkz2[~specify] = -1
            Rkz2[~specify] = -1

        Rk_out, Tk_out = k_out[Rkz2>=0], k_out[Tkz2>0]
        Tk_out[:,3] = (z_direction[Tkz2>0]*np.sqrt(Tkz2[Tkz2>0]))
        Rk_out[:,3] = (-z_direction[Rkz2>=0]*np.sqrt(Rkz2[Rkz2>=0]))
        k_out = np.vstack((Rk_out, Tk_out))

        #energy stokes vector
        if output_option and k_out.size > 0 and hasattr(self,'parameters'):
            num_R = np.sum(Rkz2>=0)
            n_out = np.hstack((n_out[Rkz2>=0], n_out[Tkz2>0]))
            n_in = np.hstack((n_in[Rkz2>=0], n_in[Tkz2>0]))
            k_in = np.vstack((k_in[Rkz2>=0],k_in[Tkz2>0]))
            matrix = self.simulator._estimate(self._k_to_rsoft(k_in))  #estimate Jones
            matrix = np.vstack((matrix[:num_R,0],matrix[num_R:,1]))
            matrix = jones_to_muller(matrix)
            k_out[:,-4:] = np.real(np.einsum('ijk,ik->ij', matrix, k_out[:,-4:]))
            if output_option == 2:  #power
                ray_k2sp = rays_tool(input_format = 'k',output_format = 'sp')
                theta_in = ray_k2sp.convert(k_in)[:,1]
                theta_out = ray_k2sp.convert(k_out)[:,1]
                power_factor = np.cos(np.deg2rad(theta_out))/np.cos(np.deg2rad(theta_in))
                k_out[:,-4:] *= (np.hstack((n_in[:num_R],n_out[num_R:]))/n_in*power_factor)[:,np.newaxis]
        
        #combine same raypath
        if k_out.size > 0:
            unique, idx, counts = np.unique(k_out[:,:-4], axis = 0, return_index=True, return_counts = True)
            overlap_rays = unique[counts>1]
            if overlap_rays.size > 0:
                k_unique = k_out[idx[counts==1]]
                if output_option:
                    stoke_vector = [np.sum(k_out[np.all(k_out[:,:-4] == k,axis = 1),-4:],axis = 0) for k in overlap_rays]
                else:
                    stoke_vector = [[1,0,0,0]]*len(overlap_rays)
                k_out = np.vstack((k_unique,np.hstack((overlap_rays, stoke_vector))))
        return k_out
    
class Fresnel_loss:
    def __init__(self, index):
        self.index = index

    def _fresnel_k(self,n_in,n_out,k_in):
        kx,ky,kz = k_in.astype(complex).T
        kz = np.abs(kz)
        SQRT = np.sqrt(n_out**2-(kx**2+ky**2))
        rs = (kz-SQRT)/(kz+SQRT)
        rp = (-n_out**2*kz+n_in**2*SQRT)/(n_out**2*kz+n_in**2*SQRT)
        ts = 2*kz/(kz+SQRT)
        tp = 2*n_out*n_in*kz/(n_out**2*kz+n_in**2*SQRT)
        zero = np.zeros_like(rp)
        r_matrix = np.array([[-rp,zero],[zero,rs]])
        t_matrix = np.array([[ tp,zero],[zero,ts]])
        return np.hstack((r_matrix,t_matrix)).T.reshape((-1,2,2,2))
    
    def launched(self,k_in,output_option = 0):
        z_direction = np.where(k_in[:,3]>=0,1,-1)
        n_out = np.where(z_direction==1,self.index[1](k_in[:,0]),self.index[0](k_in[:,0]))
        n_in = np.where(z_direction==1,self.index[0](k_in[:,0]),self.index[1](k_in[:,0]))
        Tkz2 = n_out**2-(k_in[:,1]**2+k_in[:,2]**2)
        Rkz2 = n_in**2-(k_in[:,1]**2+k_in[:,2]**2)
        Rk_out, Tk_out = k_in[Tkz2<=0], k_in[Tkz2>0]
        Tk_out[:,3] = (z_direction[Tkz2>0]*np.sqrt(Tkz2[Tkz2>0]))
        Rk_out[:,3] = (-z_direction[Tkz2<=0]*np.sqrt(Rkz2[Tkz2<=0]))
        k_out = np.vstack((Rk_out,Tk_out))

        #energy stokes vector
        if output_option and k_out.size > 0:
            num_R = np.sum(Tkz2<=0)
            n_out = np.hstack((n_out[Tkz2<=0], n_out[Tkz2>0]))
            n_in = np.hstack((n_in[Tkz2<=0], n_in[Tkz2>0]))
            k_in = np.vstack((k_in[Tkz2<=0],k_in[Tkz2>0]))
            matrix = self._fresnel_k(n_in,n_out,k_in[:,1:4])#estimate Jones
            matrix = np.vstack((matrix[:num_R,0],matrix[num_R:,1]))
            Jmatrix = jones_to_muller(matrix)
            k_out[:,-4:] = np.real(np.einsum('ijk,ik->ij', Jmatrix, k_out[:,-4:]))
            if output_option == 2:  #power
                ray_k2sp = rays_tool(input_format = 'k',output_format = 'sp')
                theta_in = ray_k2sp.convert(k_in)[:,1]
                theta_out = ray_k2sp.convert(k_out)[:,1]
                power_factor = np.cos(np.deg2rad(theta_out))/np.cos(np.deg2rad(theta_in))
                k_out[:,-4:] *= (np.hstack((n_in[:num_R],n_out[num_R:]))/n_in*power_factor)[:,np.newaxis]
        return k_out

class Receiver:
    def __init__(self):
        self.store = []

    def launched(self, k_in):
        self.store += [k_in]
        return np.empty((0, 11))

# Air_coefficient = [0,0,0,0,0,0]
# LASF46B_coefficient = [2.17988922,0.306495184,1.56882437,0.012580538,0.056719137,105.316538]    #1.9
# Air = Material('Air',Air_coefficient)
# LASF46B = Material('LASF46B',LASF46B_coefficient)
# G1 = Grating([[0.3795,11]],[Air,LASF46B],hamonics = (1,0),output_order = ('T',1,0))
# G1._set_simulator('binary_R', 'temp', [5,4], ['height','duty'], grid_size = (0.01,0.05), db_name = 'DB_binary.db')
# G1._set_structure([[0],[0]],[[0.02,0.9],[0.2,0.8]])
# F1 = Fresnel_loss([Air,LASF46B])

# k_in = np.asarray([[0.525,0,0,1,0,0,0,1,1,0,0],[0.525,0,0,1,0,0,0,1,-1,0,0]])
# a = G1.launched(k_in,output_option = 2)
# b = F1.launched(k_in)

# %%
