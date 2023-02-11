import sys
import numpy as np
import trackpy as tp
import trackpy.predict
from scipy.spatial.distance import cdist
from tqdm import tqdm_notebook as tqdm

from .utils import get_frames,get_trajectories_list
from .plots import plots

class Linker:
    '''
    Link trajectories
    '''
    def __init__(self,data,max_disp,params,memory:int=2,predict:bool=False,rn1:int=0,rn2:int=0):
        self.data           = data    
        self.directory      = params.get('directory')
        self.filename       = params.get('filename')
        self.media          = params.get('media')
        self.mode           = params.get('mode')
        self.format         = params.get('format')
        self.nframes        = params.get('nframes')
        self.diameter       = params.get('diameter')
        self.filter_size    = params.get('filter_size')
        self.bgavg          = params.get('bgavg')
        self.minbright      = params.get('minbright')
        self.masscut        = params.get('masscut')
        self.cutoff         = params.get('cutoff')
        self.initial_frame  = params.get('initial_frame')
        self.frame_rate     = params.get('frame_rate')
        self.single_frame   = params.get('single_frame')
        self.plot           = params.get('plot')
        self.max_disp       = max_disp * params['diameter']
        self.memory         = memory
        self.predict        = predict
        self.rn1            = rn1
        self.rn2            = rn2

    def generate_input_from_dict(self):
        frames  =  sorted(self.data.keys())
        coords = []
        for f in frames:
            coords.append(self.data[f][:,:2])
        return coords

    def generate_input_from_txt(self):
        frames = get_frames(self.mode,self.data)
        frame_range = np.arange(self.initial_frame,self.initial_frame + self.nframes)
        coords = []
        for f in frame_range:
            coords.append(frames[f][:,:2])
        return coords

    def link_trajectory(self, id, f_ids, links):
        if type(self.data) == dict:
            frames = sorted(self.data.keys())
        if type(self.data) == np.ndarray:
            all_frames = get_frames(self.mode,self.data)
            frame_range = np.arange(self.initial_frame,self.initial_frame + self.nframes)
            frames = [(all_frames[i]) for i in frame_range]

        data = []

        for i in f_ids:
            i = int(i)
            f_id, ids = links[i]
            if id in ids:
                id_number =  ids.index(id)
                if type(self.data) == dict:
                    data.append(self.data[frames[i]][id_number])
                if type(self.data) == np.ndarray:
                    data.append(frames[i][id_number])

        return np.vstack(data)

    def get_trajectories(self):
        @trackpy.predict.predictor
        def pred_func(t1, particle):
            velocity = np.array((-self.rn1,self.rn2))
            return particle.pos + velocity * (t1 - particle.t)

        if type(self.data) == dict:
            coords = self.generate_input_from_dict()
        if type(self.data) == np.ndarray:
            coords = self.generate_input_from_txt()
        if self.predict:
            links = tp.link_iter(coords, search_range=self.max_disp, memory = self.memory, predictor = pred_func)
        else:
            links = tp.link_iter(coords, search_range=self.max_disp, memory = self.memory)
        links = list(links)
            # ids,f_ids = [],[]
            # for i, _ids in tp.link_iter(coords, search_range=self.max_disp,memory = self.memory ):
            #     ids.extend(_ids)
            #     f_ids.append(i)

        ids, f_ids = [], []
        f_ids = []
        for i in links:
            f_ids.append(i[0])
            ids.extend(i[1])

        traj = {}
        pbar = tqdm(np.unique(np.hstack(ids)))
        for id in pbar:
            pbar.set_description('Linking')
            traj[id] = self.link_trajectory(id, f_ids, links)
        return traj

    def reconstruct(self, trajs):
        ids = trajs.keys()

        all = []
        for id in ids:
            output = np.zeros((trajs[id].shape[0], trajs[id].shape[1]))
            output= trajs[id]
            all.append(output)

        stack = np.vstack(all)
        stack = stack[stack[:,2].argsort()]
        return stack

    def return_frames_dict(self, stack):
        frames = get_frames(self.mode,stack)
        dict = {}
        for f in frames:
            if f.shape[0] != 0:
                tf = int(np.unique(f[:,2]))
                dict[tf] = f[f[:,3].argsort()]
        return dict

    def link(self):
        """
        coords: list of positions, where list index corresponds to frame number, eg., coords[frame]
        """
        if type(self.data) == dict:
            frames = sorted(self.data.keys())
            initial_particles = self.data[frames[0]].shape[0]
        if type(self.data) == np.ndarray:
            frames = get_frames(self.mode, self.data)
            if self.initial_frame >= len(frames):
                sys.exit('Text file does not contain enough frame data')
            else:
                # frame_range = np.arange(self.initial_frame,self.initial_frame + self.nframes)
                initial_particles = len(frames[self.initial_frame])

        initial_ids = np.arange(initial_particles)

        zero_ids = np.zeros((initial_particles, 1))
        zero_ids[:,0] = initial_ids
        output = {}
        if isinstance(self.data,dict):
            output[frames[0]] = np.hstack((self.data[frames[0]], zero_ids))
        elif isinstance(self.data,np.ndarray):
            output[frames[self.initial_frame][0,2]] = np.hstack((frames[self.initial_frame],zero_ids))

        if self.mode == 'single':
            return output
        else:
            if len(frames) < 3:
                sys.exit('Tracking in MULTI mode requires at least 3 frames')
            else: pass

            trjs = self.get_trajectories()

            trajs = {}
            for t in trjs:
                extended = np.zeros((trjs[t].shape[0],trjs[t].shape[1]+1))
                extended[:,:-1] = trjs[t]
                extended[:,-1] = t
                trajs[t] = extended


            if self.plot == True:
                ids = False
                plots.draw_trajs(trajs,ids)
            output = self.return_frames_dict(self.reconstruct(trajs))
            return output,trajs

class Filtering:
    def __init__(self,data,params,cutoff,ids:bool=None):
        self.data           = data
        self.mode           = params['mode']
        self.cutoff         = cutoff * params['diameter']
        self.initial_frame  = params['initial_frame']
        self.plot           = params['plot']
        self.ids            = ids

    def filter_particles(self):
        if self.mode == 'single':
            sys.exit('Cannot filter pinned particles in single mode')
        else:
            if isinstance(self.data,np.ndarray):
                frame_range = np.arange(self.initial_frame,self.initial_frame + self.nframes)
                all_frames = get_frames(self.mode,self.data)
                frames = [(all_frames[i]) for i in range(len(frame_range))]
                trajs = get_trajectories_list(frames)

            elif isinstance(self.data,dict):
                trajs = get_trajectories_list(self.data)

            filtered = []
            pbar = tqdm(trajs)

            for t in pbar:
                pbar.set_description('Filtering')
                if t.shape[0] > 0:
                    disp = cdist(t[0,:2].reshape(1,2), t[-1,:2].reshape(1,2))
                    if disp < self.cutoff:
                        pass
                    else:
                        filtered.append(t)

            print ('{} trajectories removed'.format(len(trajs) - len(filtered)))

            trajs = {}
            for i in filtered:
                trajs[i[0,3]] = np.vstack(i)
        if self.plot ==  True:
            plots.draw_trajs(trajs,self.ids)
        return filtered, trajs

    def filter(self):
            filtered, trajs = self.filter_particles()
            stack = np.vstack(filtered)
            frames = get_frames(self.mode, stack)

            output = {}
            for f in frames:
                if f.shape[0] != 0:
                    output[int(f[0,2])] = np.vstack(f)
            return output,trajs
