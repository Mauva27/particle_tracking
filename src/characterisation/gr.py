import sys
import numpy as np
import pylab as pl
from tqdm import tqdm_notebook as tqdm
from scipy.spatial.distance import pdist
from scipy.stats import binned_statistic
# from scipy.interpolate import spline, interp1d
from scipy.optimize import curve_fit
# from peakutils import indexes
from ..utils import get_frames
import ipdb

def powerlaw_decay(d,a,n,c):
    return a*(d**(-n))+c

def exponential_decay(d,a,e,c):
    return a*(np.exp((-d)/e)) + c

class get_gr:
    def __init__(self,data, diameter, gr_nframes,dist_cutoff, plot, fit, kind):
        self.data = data
        self.diameter = diameter
        self.nframes = gr_nframes
        self.cutoff = dist_cutoff
        self.plot = plot
        self.fit = fit
        self.kind = kind

    def fit_gr(self, data):
        r = data[:,0]
        gr = data[:,1]

        # peaks = indexes(gr ,thres=0.5/gr.max(),min_dist=self.diameter)
        pl.figure(figsize = (5,4), num = 'Pair correlation function')
        pl.plot(r, gr)

        if self.kind == 'power_law':
            popt,pcov=curve_fit(powerlaw_decay,r[peaks],gr[peaks])
            pl.plot(r, power_law(r, *popt), label = '$\eta$ = {}'.format(popt[1]))
        if self.kind == 'exponential':
            popt,pcov=curve_fit(exponential_decay,r[peaks],gr[peaks])
            pl.plot(r, exponential_decay(r, *popt), label = '$ \\xi ${}'.format(popt[1]))
        pl.xlabel('r / $\sigma$')
        pl.ylabel('g(r)')
        pl.legend()

    def compute_gr(self, data, rand_cutoff):
        radius =  self.diameter / 2.

        coords = data[:,:2]
        coords = coords[coords[:,0] < coords[:,0].max() - radius]
        coords = coords[coords[:,0] > coords[:,0].min() + radius]
        coords = coords[coords[:,1] < coords[:,1].max() - radius]
        coords = coords[coords[:,1] > coords[:,1].min() + radius]

        n = coords.shape[0]
        bins = np.arange(0, coords[:,0].max(), 1)

        # Generate random positions for normalisation

        output = np.zeros((self.cutoff, 2))

        for i in range(self.cutoff):
            all_rand_hist = 0
            for j in range(rand_cutoff):
                rands = np.zeros((n,2))
                x_rand = np.random.uniform(coords[:,0].min(), coords[:,0].max(), n)
                y_rand = np.random.uniform(coords[:,1].min(), coords[:,1].max(), n)
                rands[:,0] = x_rand
                rands[:,1] = y_rand

                rand_dists = pdist(rands, 'euclidean')
                rand_hist,_,_ = binned_statistic(rand_dists, rand_dists, statistic = 'count', bins = bins)
                all_rand_hist += rand_hist
            all_rand_hist /= j

            dists = pdist(coords, 'euclidean')
            hist, bins,_ = binned_statistic(dists, dists, statistic = 'count', bins =  bins)

            bincenters = 0.5 * (bins[1:] + bins[:-1])
            gr_hist = hist/all_rand_hist
            gr_hist[np.isnan(gr_hist)] = 0
            gr_hist[np.isinf(gr_hist)] = 0
            bincenters = bincenters[0:self.cutoff]
            gr_hist = gr_hist[0:self.cutoff]

            output[:,0] = bincenters
            output[:,1] += gr_hist

        output[:,0] /= self.diameter
        output[:,1] /= self.cutoff
        return output

    def gor(self):
        if  type(self.data) ==  dict:
            frames = self.data.keys()
        if type(self.data) == np.ndarray:
            mode = 'multi'
            frames = get_frames(mode,self.data)

        if self.nframes > len(frames):
            sys.exit('nframes exceed number of frames in data')
        else:
            pbar = tqdm(range(self.nframes))
            gr_out = np.zeros((self.cutoff, 2))
            for f in pbar:
                pbar.set_description('Progress of g(r). Frame {}'.format(frames[f]))
                if type(self.data) ==  dict:
                    gr = self.compute_gr(self.data[frames[f]], rand_cutoff = 5)
                if type(self.data) == np.ndarray:
                    gr = self.compute_gr(frames[f], rand_cutoff  = 5)
                gr_out[:,0] = gr[:,0]
                gr_out[:,1] += gr[:,1]
            gr_out[:,1] /= self.nframes

            # r = np.linspace(gr_out[:,0].min(), gr_out[:,0].max(), self.cutoff * 2.)
            # linear = interp1d(gr_out[:,0], gr_out[:,1], kind = 'linear')
            # linear_gr = linear(r)

            # r_new = np.linspace(gr_out[:,0].min(), gr_out[:,0].max(), 120)
            # cubic = interp1d(r, linear_gr, kind = 'cubic')
            # soft_gr = cubic(r_new)
            # soft_gr[r_new<0.8] = 0
            # soft_gr[soft_gr < 0] = 0

            new_gr_out = np.zeros((r_new.shape[0], 2))
            new_gr_out[:,0] = r_new
            new_gr_out[:,1] = soft_gr

            if self.plot and not self.fit:
                pl.figure(figsize = (5,4), num = 'Pair Correlation Function')
                pl.plot(new_gr_out[:,0], new_gr_out[:,1])
                pl.xlabel('r / $\sigma$')
                pl.ylabel('g(r)')
            if self.fit:
                self.fit_gr(new_gr_out)
        return gr_out,new_gr_out
