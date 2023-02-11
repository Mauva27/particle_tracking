import os
import numpy as np
import pickle

from PIL import Image
from tqdm import tqdm_notebook as tqdm
import math

def open_pickle(file):
    with open(file, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p

def dot_product(v0,v1):
    return sum(np.dot(i,j) for i,j  in zip(v0,v1))

def vector_magnitude(v):
    return math.sqrt(dot_product(v,v))

def  get_angle(v0,v1):
    if dot_product(v0,v1) / (vector_magnitude(v0) * vector_magnitude(v1)) == -1.0000000000000002:
        return math.acos(-1)
    else:
        return math.acos(dot_product(v0,v1) / (vector_magnitude(v0) * vector_magnitude(v1)))

def to_polar_coords(data:np.ndarray):
    '''
    Transforms euclidean coords into polar coords
    '''
    pcoords = np.zeros((data.shape[0],2))
    pcoords[:,0] = np.sqrt(data[:,0]**2 + data[:,1]**2)
    for i in range(data.shape[0]):
        angle_theta = np.abs(np.arctan(data[i,1] / data[i,0]))
        theta = 0.0
        if data[i,0] == 0 and data[i,1] > 0: theta = np.pi/2.0
        elif data[i,0] ==0 and data[i,1] < 0: theta = 3.0*np.pi / 2.0
        elif data[i,0] > 0 and data[i,1] ==0: theta = 0
        elif data[i,0] < 0 and data[i,1] == 0: theta = np.pi
        elif data[i,0] > 0 and data[i,1] > 0: theta = angle_theta
        elif data[i,0] < 0 and data[i,1] > 0: theta = np.pi - angle_theta
        elif data[i,0] < 0 and data[i,1] < 0: theta = np.pi + angle_theta
        elif data[i,0] > 0 and data[i,1] < 0: theta = 2.0*np.pi - angle_theta
        pcoords[i,1] = theta
    return pcoords

def get_centre_of_mass(coords):
    return np.array([np.mean(coords[:,0]), np.mean(coords[:,1])])

def load_image(directory, prefix, frame, fmt):
    '''
    Imports frame
    '''
    img = Image.open(directory + prefix + '{:05d}'.format(frame) + fmt, mode = 'r')
    return np.array(img.copy())

def get_frames(mode,data):
    if mode == 'single':
        frame_data = [data]
        return frame_data
    else:
        min_frame = int(data[:,2].min())
        max_frame = int(data[:,2].max())
        frame_data = [data[data[:,2] == x] for x in range(min_frame,max_frame + 1,1)]
        return frame_data

def get_trajectories_list(data):
    if type(data) == dict:
        frames = sorted(data.keys())
        all = []
        for f in frames:
            all.append(data[f])

        stack = np.vstack(all)
    if type(data) == np.ndarray:
        mode = 'multi'
        frames = get_frames(mode,data)
        all = []
        for f in frames:
            all.append(f)

        stack = np.vstack(all)
    if type(data) == list:
        stack = np.vstack(data)

    trajs = [stack[stack[:,3] == i] for i in range(int(stack[:,3].max()) + 1)]
    return trajs

def create_folder(directory):
    if not os.path.exists(directory + 'data'):
        os.mkdir(directory + 'data')
        print ("data directory has been created")

def create_traj_folder(directory):
    if not os.path.exists(directory + 'trajectories'):
        os.mkdir(directory + 'trajectories')
        print ("trajectories directory has been created")

def saving_pickle(directory,filename,data):
    create_folder(directory)
    pickle.dump(data,open(directory + 'data/{}.p'.format(filename), 'wb'))

def saving_xyz(directory,filename,data, frame_rate):
    create_folder(directory)
    # ipdb.set_trace()
    f = open(directory + 'data/' + '{}.xyz'.format(filename), 'w')
    f.close()

    if len(data) == 1:
        for f in data:
            if data[f].shape[1] == 2:
                with open(directory + 'data/' + '{}.xyz'.format(filename), 'a') as file:
                    file.write('{}\nProperties=species:S:1:pos:R:2 Time=0\n'.format(data[f].shape[0]))
                output = np.zeros((data[f].shape[0],2))
                output[:,0] = data[f][:,0]
                output[:,1] = data[f][:,1]
                for row in output:
                    re = row.reshape(1,2)
                    with open(directory + 'data/' +'{}.xyz'.format(filename), 'a') as this_row:
                        np.savetxt(this_row, re, delimiter = '\t', fmt = ' '.join(['A\t%1.5f'] + ['%1.5f']))

    else:
        frames = data.keys()

        for i, f in enumerate(frames):
            time = float(f) / float(frame_rate)
            # Features data
            if data[f].shape[1] == 3:
                with open(directory + 'data/' + '{}.xyz'.format(filename), 'a') as file:
                    file.write('{}\nProperties=species:S:1:pos:R:2 Time={}\n'.format(data[f].shape[0], time))
                output = np.zeros((data[f].shape[0],2))
                output[:,0] = data[f][:,0]
                output[:,1] = data[f][:,1]
                for row in output:
                    re = row.reshape(1,2)
                    with open(directory + 'data/' +'{}.xyz'.format(filename), 'a') as this_row:
                        np.savetxt(this_row, re, delimiter = '\t', fmt = ' '.join(['A\t%1.5f'] + ['%1.5f']))
            # Trajectory data
            if data[f].shape[1] == 4:
                with open(directory + 'data/' + '{}.xyz'.format(filename), 'a') as file:
                    file.write('{}\nProperties=species:S:1:pos:R:2:id:I:1 Time={}\n'.format(data[f].shape[0], time))
                output = np.zeros((data[f].shape[0],4))
                output[:,0] = data[f][:,0]
                output[:,1] = data[f][:,1]
                output[:,2] = data[f][:,2]
                output[:,3] = data[f][:,3]
                for row in output:
                    re = row.reshape(1,4)
                    with open(directory + 'data/' +'{}.xyz'.format(filename), 'a') as this_row:
                        np.savetxt(this_row, re, delimiter = '\t', fmt = ' '.join(['A\t%1.5f'] + ['%1.5f']*3))
            if data[f].shape[1] == 5:
                with open(directory + 'data/' + '{}.xyz'.format(filename), 'a') as file:
                    file.write('{}\nProperties=species:S:1:pos:R:2:id:I:1:coordination:Z:1 Time={}\n'.format(data[f].shape[0], time))
                output = np.zeros((data[f].shape[0],4))
                output[:,0] = data[f][:,0]
                output[:,1] = data[f][:,1]
                output[:,2] = data[f][:,3]
                output[:,3] = data[f][:,-1]

                for row in output:
                    re = row.reshape(1,4)
                    with open(directory + 'data/' +'{}.xyz'.format(filename), 'a') as this_row:
                        np.savetxt(this_row, re, delimiter = '\t', fmt = ' '.join(['A\t%1.5f'] + ['%1.5f']*(output.shape[1] - 1)))
            # Boop features
            elif data[f].shape[1] == 7:
                with open(directory + 'data/' + '{}.xyz'.format(filename), 'a') as file:
                    file.write('{}\nProperties=species:S:1:pos:R:2:boop:B:1 Time={}\n'.format(data[f].shape[0], time))
                output = np.zeros((data[f].shape[0], 4))
                output[:,0] = data[f][:,0]
                output[:,1] = data[f][:,1]
                output[:,2] = data[f][:,-2]
                output[:,3] = data[f][:,-1]

                for row in output:
                    re = row.reshape(1,4)
                    with open(directory +'data/' + '{}.xyz'.format(filename), 'a') as this_row:
                        np.savetxt(this_row, re, delimiter = '\t', fmt =  '  '.join(['A\t%1.5f'] + ['%1.5f'] * (output.shape[1] - 1)))
            # Boop trajectories
            elif data[f].shape[1] == 8:
                with open(directory + 'data/' + '{}.xyz'.format(filename), 'a') as file:
                    file.write('{}\nProperties=species:S:1:pos:R:2:id:I:1:boop:B:1 Time={}\n'.format(data[f].shape[0], time))
                output = np.zeros((data[f].shape[0], 5))
                output[:,0] = data[f][:,0]
                output[:,1] = data[f][:,1]
                output[:,2] = data[f][:,3]
                output[:,3] = data[f][:,-2]
                output[:,4] = data[f][:,-1]
                for row in output:
                    re = row.reshape(1,5)
                    with open(directory +'data/' + '{}.xyz'.format(filename), 'a') as this_row:
                        np.savetxt(this_row, re, delimiter = '\t', fmt =  '  '.join(['A\t%1.5f'] + ['%1.5f'] * (output.shape[1] - 1)))
            # Boop extended
            elif data[f].shape[1] == 10:
                with open(directory + 'data/' + '{}.xyz'.format(filename), 'a') as file:
                    file.write('{}\nProperties=species:S:1:pos:R:2:id:I:1:vel:V:2:boop:B:1 Time={}\n'.format(data[f].shape[0], time))
                output = np.zeros((data[f].shape[0], 7))
                output[:,0] = data[f][:,0]
                output[:,1] = data[f][:,1]
                output[:,2] = data[f][:,3]
                output[:,3] = data[f][:,4]
                output[:,4] = data[f][:,5]
                output[:,5] = data[f][:,-2]
                output[:,6] = data[f][:,-1]
                for row in output:
                    re = row.reshape(1,7)
                    with open(directory +'data/' + '{}.xyz'.format(filename), 'a') as this_row:
                        np.savetxt(this_row, re, delimiter = '\t', fmt =  '  '.join(['A\t%1.5f'] + ['%1.5f'] * (output.shape[1] - 1)))


def saving_txt(directory, filename, data):
    create_folder(directory)
    f = open(directory + 'data/' + '{}.txt'.format(filename), 'w')
    f.close()

    if type(data) == dict:
        frames = sorted(data.keys())
        if data[frames[0]].shape[1] >= 3:
            for fr in frames:
                output = np.zeros((data[fr].shape[0], data[fr].shape[1]))
                output[:] = data[fr]
                with open(directory +'data/'+ '{}.txt'.format(filename), 'a') as file:
                    np.savetxt(file, output, delimiter = '\t', fmt =  '  '.join(['%1.5f']))
    else:
        with open(directory+'data/{}.txt'.format(filename),'a') as file:
            np.savetxt(file,data,fmt = '%1.5f')

def saving(directory,data,format,filename,frame_rate=0):
    if format == 'all':
        saving_xyz(directory,filename,data,frame_rate)
        saving_txt(directory,filename,data)
        saving_pickle(directory,filename,data)
    elif format == 'pickle/xyz':
        saving_pickle(directory,filename,data)
        saving_xyz(directory,filename,data,frame_rate)
    elif format == 'pickle/txt':
        saving_pickle(directory,filename,data)
        saving_txt(directory,filename,data)
    elif format == 'txt/xyz':
        saving_txt(directory,filename,data)
        saving_xyz(directory,filename,data,frame_rate)
    elif format == 'pickle':
        saving_pickle(directory,filename,data)
    elif format == 'xyz':
        saving_xyz(directory,filename,data,frame_rate)
    elif format == 'txt':
        saving_txt(directory,filename,data)

# def export_imgs(directory, prefix, fmt,data,trajs):
#     create_traj_folder(directory)

#     frames = sorted(data.keys())
#     ids_in_frames = [data[f][:,3] for f in frames]

#     pl.close()
#     fig,ax = pl.subplots(figsize=(3,3))
#     pbar = tqdm(enumerate(frames))
#     for i,t in pbar:
#         pbar.set_description('Exporting')
#         frame = int(t)
#         img = load_image(directory,prefix,frame,fmt)
#         ax.imshow(img,cmap = 'gray')

#         for id in ids_in_frames[i]:
#             ax.plot(trajs[id][:i,0], trajs[id][:i,1], lw = 0.5, alpha = 0.75)

#         pl.axis('off')
#         pl.savefig(directory+'trajectories/t_'+'{:05d}'.format(frame)+fmt)

