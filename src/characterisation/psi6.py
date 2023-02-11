import cmath
import pickle
import numpy as np

from pyvoro import compute_2d_voronoi
from tqdm import tqdm_notebook as tqdm

from ..utils import get_frames, get_trajectories_list
from ..plots import plots

'''
Something to add to documentation:

psi6.py needs pyvoro to compute voronoi tesselation


To change:
Instead of reading coords, lets read raw_coordds with particle index and export according this id
'''

class Psi6:
    def __init__(self,data,diameter,mode:str='static',plot:bool=True):
        assert isinstance(data,dict); "Input data must be a dictionary"
        self.data       = data
        self.nframes    = data.keys()
        self.mode       = mode
        self.diameter   = diameter
        self.dist_cut   = diameter * 1.1
        self.plot       = plot

    def get_neighs(self, coords):
        max_x = coords[:,0].max() + self.diameter
        max_y = coords[:,1].max() + self.diameter

        cells =  compute_2d_voronoi(coords,[[0, max_x],[0,max_y]], 1.0)
        num_neigh = []
        which_neighs = []
        for cell in cells:
            neigh = 0
            this_cell_adj = []
            for adjacent in cell['faces']:
                if adjacent['adjacent_cell'] >= 0:
                    dist = np.linalg.norm(cell['original']-cells[adjacent['adjacent_cell']]['original'])
                if adjacent['adjacent_cell'] >= 0 and dist < self.dist_cut:
                    neigh += 1
                    a = adjacent['adjacent_cell']
                # else:
                    # a = 0
                    this_cell_adj.append(a)
            which_neighs.append(this_cell_adj)
            num_neigh.append(neigh)

        real_neighs = []
        for i in range(len(which_neighs)):
            this_neigh = np.array(which_neighs[i])
            real_neighs.append(this_neigh)
        return cells, num_neigh, real_neighs

    def compute_boop(self, coords):
        cells, n_neighs, neighs = self.get_neighs(coords)

        boop = np.zeros((len(cells), 4))

        for k in range(boop.shape[0]):
            k = int(k)
            this_neigh = neighs[k]
            neigh_disp = np.zeros((this_neigh.shape[0], 4))

            cos_total = 0
            sin_total = 0

            for j in range(this_neigh.shape[0]):
                neigh_disp[j,0] = coords[this_neigh[j],0] - coords[k,0]

                neigh_disp[j,1] = coords[this_neigh[j],1] - coords[k,1]

                neigh_disp[j,2] = np.sqrt(neigh_disp[j,0]**2 + neigh_disp[j,1]**2)

                angle_theta = np.arctan(np.abs(neigh_disp[j,1]/neigh_disp[j,0]))

                theta = 0.0
                if neigh_disp[j,0] == 0 and neigh_disp[j,1] > 0: theta = np.pi/2.0
                elif neigh_disp[j,0] ==0 and neigh_disp[j,1] < 0: theta = 3.0*np.pi / 2.0
                elif neigh_disp[j,0] > 0 and neigh_disp[j,1] ==0: theta = 0
                elif neigh_disp[j,0] < 0 and neigh_disp[j,1] == 0: theta = np.pi
                elif neigh_disp[j,0] > 0 and neigh_disp[j,1] > 0: theta = angle_theta
                elif neigh_disp[j,0] < 0 and neigh_disp[j,1] > 0: theta = np.pi - angle_theta
                elif neigh_disp[j,0] < 0 and neigh_disp[j,1] < 0: theta = np.pi + angle_theta
                elif neigh_disp[j,0] > 0 and neigh_disp[j,1] < 0: theta = 2.0*np.pi - angle_theta

                neigh_disp[j,3] = theta
                cos_total = cos_total + np.cos(6.0*theta)
                sin_total = sin_total + np.sin(6.0*theta)

            if not this_neigh.shape[0]:
                boop[k,0] = 0
                boop[k,1] = 0
                boop[k,2] = 0
                boop[k,3] = np.pi/6.
            else:
                boop[k,0] = cos_total / this_neigh.shape[0]
                boop[k,1] = sin_total / this_neigh.shape[0]
                boop[k,2] = np.sqrt(boop[k,0]**2 + boop[k,1]**2)
                c = complex(boop[k,0] ,boop[k,1])
                phase = cmath.phase(c)+np.pi
                # if phase >0:
                #     phase += np.pi
                # if phase < 0:
                #     phase += np.pi
                boop[k,3] = phase/6.
        return boop

    def get_boop(self):
        boop = {}
        if self.mode == 'static':
            if type(self.data) == dict:
                frames = sorted(self.data.keys())
                pbar = tqdm(range(frames[0],frames[-1]+1,1))
                for f in pbar:
                    pbar.set_description('Computing BOOP. Frame: {} '.format(f+2))
                    psi6 = self.compute_boop(self.data[f][:,:2])
                    boop[f] = np.hstack((self.data[f], psi6))

            if self.plot == True:
                plots.plot_boop(boop)
            return boop

        if self.mode == 'dynamic':
            if type(self.data) == dict:
                frames = sorted(self.data.keys())
                pbar = tqdm(range(frames[0],frames[-1]+1,1))

                for f in pbar:
                    pbar.set_description('Computing BOOP. Frame: {} '.format(f+2))
                    psi6 = self.compute_boop(self.data[f][:,:2])
                    boop[f] = np.hstack((self.data[f], psi6))

            trajs = get_trajectories_list(boop)
            btrajs = {}
            for i,j in enumerate(trajs):
                btrajs[i] = j

            if self.plot == True:
                plots.plot_boop(boop)

            return boop, btrajs


class X6:
    def __init__(self,input,directory,data,params,save):
        self.input = input
        self.directory = directory
        self.mode = params['mode']
        self.ws = params['which_single']
        self.initial_frame = params['initial_frame']
        if self.input == 'Dictionary':
            self.data = data
            self.nframes = params['nframes']
        if self.input == 'Text file':
            self.data = np.loadtxt(self.directory + 'data/psi6.txt')
            self.nframes = self.data[:,2].max() +1
        self.save = save

    def compute_x6(self,boop):
        return np.mean(boop**2) - np.mean(boop)**2

    def mean_x6(self):
        x6 = 0

        if type(self.data) == dict:
            frames = sorted(self.data.keys())
            for f in frames:
                x6 += self.compute_x6(self.data[f][:,-1])
            x6 /= len(frames)
        if type(self.data) == np.ndarray:
            frames = get_frames(self.mode,self.data)
            for f in frames:
                x6 += self.compute_x6(f[:,-1])
            x6 /= len(frames)

        x6 = np.array(x6).reshape(1,)


        if self.save == True:
            np.savetxt(self.directory + 'data/chi6.txt',x6,fmt = '%1.5f')
