import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
# import matplotlib.animation as animation
from .__quichi__ import __quichi__
from ..read import load_image, load_movie_frames
from . import sans_serif

def over_plot(media,mode,directory,nframes,filename,single_frame,format,data,ny,frame_rate):
    '''
    shosingle_frame all the features found in pretrack
    '''
    pl.close()
    if mode == 'single':
        frame = data.keys()
        if media == 'Images':
            img = load_image(directory, filename,single_frame,format)
        if media == "Movie":
            img = load_movie_frames(directory, filename,single_frame,format)
        pl.figure(figsize = (5,5), num='Tracked particles')
        pl.imshow(img, cmap = 'gray')
        pl.plot(data[single_frame][:,0], -data[single_frame][:,1]+ny, lw = 0, marker = 'o', ms = 5, mec = 'r', mfc = 'None', zorder = 1)
        pl.xticks([])
        pl.yticks([])
        pl.xlabel('{} particles'.format(data[single_frame].shape[0]))
    if mode == 'multi':
        fk = sorted(data.keys())

        if (nframes > 1) & (nframes <= 20):
            cols = 5
            which = nframes / cols
            rang = np.ceil(np.linspace(fk[0],fk[-1],cols))
            fig, axs = pl.subplots(1,len(rang), figsize=(10,3), num='Tracked particles')
            axs = axs.ravel()

            for i, f in enumerate(rang):
                i,f = int(i), int(f)
                if media == 'Images':
                    img = load_image(directory,filename,f,format)
                if media == 'Movie':
                    img = load_movie_frames(directory,filename,f,format)
                axs[i].imshow(img, cmap = 'gray')
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].plot(data[f][:,0], -data[f][:,1]+ny, lw = 0,ms = 3, marker = 'o', mec = 'r', mfc = 'None', zorder = 1)
                axs[i].set_title('t = {:0.3f} s'. format(f / float(frame_rate)))
                axs[i].set_xlabel('{} particles'.format(data[f].shape[0]))
        elif (nframes > 20):
            cols = 5
            which = nframes / cols
            rang = np.ceil(np.linspace(fk[0],fk[-1],cols))
            fig, axs = pl.subplots(1,len(rang), figsize=(10,3), num='Tracked particles')
            axs = axs.ravel()
            for i,f in enumerate(rang):
                f = int(f)
                if media == 'Images':
                    img = load_image(directory,filename,f,format)
                if media == 'Movie':
                    img = load_movie_frames(directory,filename,f,format)
                axs[i].imshow(img, cmap = 'gray')
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].plot(data[f][:,0], -data[f][:,1]+ny, lw = 0,ms = 3, marker = 'o', mec = 'r', mfc = 'None')
                axs[i].set_title('t = {:0.2f} s'. format(f / float(frame_rate)))
                axs[i].set_xlabel('{} particles'.format(data[f].shape[0]))

def plot_clusters(mode,nframes,data,ids):
    frames = sorted(data.keys())
    pl.close()
    if mode == 'single':
        fig = pl.figure(figsize = (4,4),num = 'Clusters in frame {}'.format(frames[0]))
        ax = fig.gca()
        for ckey in data[frames[0]]:
            ax.scatter(data[frames[0]][ckey]['coords'][:,0], data[frames[0]][ckey]['coords'][:,1], s = 3, alpha = 0.25)
            if ids:
                ax.text(data[frames[0]][ckey]['com'][0],data[frames[0]][ckey]['com'][1],'{}'.format(ckey), fontsize = 12)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    elif mode == 'multi':
        fk = sorted(data.keys())
        if (nframes > 1) & (nframes <= 20):
            cols = int(nframes // (nframes*0.25))
            which = nframes / cols
            rang = np.ceil(np.linspace(fk[0],fk[-1],cols))
            fig, axs = pl.subplots(1,len(rang), figsize=(12,3.5), num='Clusters')
            for i, f in enumerate(rang):
                i,f = int(i), int(f)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                for ckey in data[f]:
                    axs[i].scatter(data[f][ckey]['coords'][:,0], data[f][ckey]['coords'][:,1], s = 1, alpha = 0.25)
                    if ids:
                        axs[i].text(data[f][ckey]['com'][0],data[f][ckey]['com'][1],'{}'.format(ckey), fontsize = 12)
                    axs[i].set_xlabel('Frame {}'.format(f))
        elif (nframes > 20):
            cols = int(nframes // (nframes*0.25))
            which = nframes / cols
            rang = np.ceil(np.linspace(fk[0],fk[-1],cols))
            fig, axs = pl.subplots(1,len(rang), figsize=(13,2.5), num='Tracked particles')
            axs = axs.ravel()
            for i,f in enumerate(rang):
                f = int(f)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                for ckey in data[f]:
                    axs[i].scatter(data[f][ckey]['coords'][:,0], data[f][ckey]['coords'][:,1], s = 1, alpha = 0.25)
                    if ids:
                        axs[i].text(data[f][ckey]['com'][0],data[f][ckey]['com'][1],'{}'.format(ckey), fontsize = 12)
                    axs[i].set_xlabel('Frame {}'.format(f))


def plot_interface(data):
    fig,axs = pl.subplots(1,5,figsize=(13,3), num='Interface')
    axs = axs.ravel()

    clusters = sorted(data.keys())

    for i in np.arange(5):
        axs[i].scatter(data[clusters[i]]['interior'][:,0], data[clusters[i]]['interior'][:,1], s=5, color = 'b')
        axs[i].scatter(data[clusters[i]]['boundary'][:,0], data[clusters[i]]['boundary'][:,1], s=5, color = 'm')
        axs[i]. set_xticks([])
        axs[i]. set_yticks([])


def plot_boop(data):
    '''
    scatters psi6 for every particle
    '''
    pl.close()
    cmap = mpl.colors.ListedColormap(__quichi__, '__quichi__')

    fk = sorted(data.keys())
    nframes = len(fk)
    if (nframes > 1) & (nframes <= 20):
        cols = int(nframes // (nframes*0.25))
        which = nframes / cols
        rang = np.ceil(np.linspace(fk[0],fk[-1],cols))
        fig, axs = pl.subplots(1,len(rang), figsize=(12,4.5), num='BOOP in tracked particles')

        for i, f in enumerate(rang):
            i,f = int(i), int(f)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            sc = axs[i].scatter(data[f][:,0], data[f][:,1] , c = data[f][:,-2], cmap = cmap)
            axs[i].set_xlabel('{} particles'.format(data[f].shape[0]))
            cbar = pl.colorbar(sc, ax = axs[i], orientation='horizontal', shrink = 0.7, aspect = 10, pad = 0.12, ticks = [0.2,0.4,0.6,0.8])
    elif (nframes > 20):
        cols = int(nframes // (nframes*0.25))
        which = nframes / cols
        rang = np.ceil(np.linspace(fk[0],fk[-1],cols))
        fig, axs = pl.subplots(1,len(rang), figsize=(13,4.5), num='Tracked particles')
        axs = axs.ravel()
        for i,f in enumerate(rang):
            f = int(f)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            sc = axs[i].scatter(data[f][:,0], data[f][:,1], c = data[f][:,-2], cmap = cmap)
            axs[i].set_xlabel('{} particles'.format(data[f].shape[0]))
            cbar = pl.colorbar(sc, ax = axs[i], orientation='horizontal', shrink = 0.7, aspect = 10, pad = 0.12, ticks = [0.2,0.4,0.6,0.8])

def draw_trajs(data,ids):
    '''
    draw in different colors the linked trajectories
    '''
    pl.close()
    fig,ax = pl.subplots(figsize = (4,4), num = 'Trajectories')
    for t in data:
        ax.plot(data[t][:,0], data[t][:,1], alpha = 0.75,label = t)
    ax.set_xticks([])
    ax.set_yticks([])
    pl.axis('off')
    if ids:
        pl.legend()

def quivers(data):
    frames = data.keys()
    cmap = mpl.colors.ListedColormap(__quichi__, '__quichi__')
    norm = mpl.colors.Normalize(vmin=frames[0],vmax=frames[-1])
    cm = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cm.set_array([])

    alpha = np.linspace(0.1,1,len(frames))

    pl.close()
    pl.figure(figsize = (5,5))

    for i,f in enumerate(frames):
        pl.quiver(data[f][:,0],data[f][:,1],data[f][:,-2],data[f][:,-1], color = cm.to_rgba(f), alpha = alpha[i])

def plot_dynamics(exp_data,sim_data,dimensions,lw,ticks,path,filename,scale=None,labels=[None,None],range=[None,None],save=None):
    cmap = mpl.colors.ListedColormap(__quichi__[::-1], '__quichi__')
    cnorm = mpl.colors.Normalize(vmin=0,vmax=len(data[phi].keys()))
    cexp = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cexp.set_array([])

    markers = ['o', 's', 'p', 'h', 'D', 'v', '^', '*', 'X', 'P', 'H', '>', 'd']

    fig = pl.figure(figsize = dimensions)
    ax = fig.gca()

    for i,ephi in enumerate(exp_data):
        for j,epe in enumerate(sorted(exp_data[ephi])):
            ax.plot(exp[ephi][epe][:,0],exp[ephi][epe][:,1],lw = lw,marker = markers[i])
