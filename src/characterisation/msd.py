from __future__ import division
import sys
import numpy as np
from scipy.optimize import curve_fit
from ..utils import get_trajectories_list
import pylab as pl
import ipdb

class MSD:
    def __init__(self,input,directory,data,params,plot,fit,kind,points,tau,velocity,references,save):
        self.input = input
        self.directory = directory
        self.diameter = params['diameter']
        if self.input == 'Dictionary':
            self.data = data
            self.nframes = params['nframes']
        if self.input == 'Pickle':
            self.data = pickle.load(open(self.directory + 'data/psi6.p', 'rb'))
            self.nframes = len(self.data.keys())
        if self.input == 'Text file':
            self.data = np.loadtxt(self.directory+'data/filtered_coords.txt')
            self.nframes = self.data[:,2].max() + 1
        self.frame_rate = params['frame_rate']
        self.plot = plot
        self.points = points
        self.fit = fit
        self.kind = kind
        self.velocity = velocity
        self.tau = tau
        self.references = references
        self.save = save
        # ipdb.set_trace()

    def passive(self,t,Dt):
        return 4*Dt*t

    def ABP(self,t,Dt,tau,v):
        Dt = 0.05380457070348083
        # tau = 0.183
        return 4 * Dt* t + ((3 ** -1) * (v ** 2) * (tau ** 2)) * (2 * (t / tau) + np.exp(-2 * t / tau) - 1)

    def quincke(self,t,tau,v):
        return 2 * (v ** 2) * (tau ** 2) * ((t / tau) - 1 + np.exp(-t / tau))

    def select_power_of_two(self,n):
        current_exp = int(np.ceil(np.log2(n+1)))
        if n == 2**current_exp:
            n_fft = n
        if n < 2**current_exp:
            n_fft = 2**current_exp
        elif n > 2**current_exp:
            n_fft = 2**(current_exp+1)
        return n_fft

    def autocorrelation_1d(self,pos):
        N = len(pos)
        n_fft = self.select_power_of_two(N)

        R_data = np.zeros(2*n_fft)
        R_data[:N] = pos

        F_data = np.fft.fft(R_data)

        result = np.fft.ifft(F_data*F_data.conj())[:N].real/(N-np.arange(N))
        return result[:N]

    def compute_msd(self,coords):
        coords = np.asarray(coords)
        if coords.ndim==1:
            coords = coords.reshape((-1,1))
        N = len(coords)
        rsq = np.sum(coords**2, axis=1)
        MSD = np.zeros(N, dtype=float)

        SAB = self.autocorrelation_1d(coords[:,0])
        for i in range(1, coords.shape[1]):
            SAB += self.autocorrelation_1d(coords[:,i])

        SUMSQ = 2*np.sum(rsq)

        m = 0
        MSD[m] = SUMSQ - 2*SAB[m]*N

        MSD[1:] = (SUMSQ - np.cumsum(rsq)[:-1] - np.cumsum(rsq[1:][::-1])) / (N-1-np.arange(N-1))
        MSD[1:] -= 2*SAB[1:]
        return MSD

    def msd_output(self):
        all = []
        if type(self.data) == dict:
            trajs = sorted(self.data.keys())
            ntrajs = len(trajs)

            for t in trajs:
                t = int(t)
                all.append(self.compute_msd(self.data[t][:,:2]))

        if type(self.data) == np.ndarray:
            trajs = get_trajectories_list(self.data)
            ntrajs = 0
            for t in trajs:
                if not t.shape[0] == 0:
                    all.append(self.compute_msd(t[:,:2]))
                    ntrajs += 1
                else:
                    pass
        msd_matrix = np.zeros((int(self.nframes),int(ntrajs)))

        for i, j in enumerate(all):
            msd_matrix[:j.shape[0],i] = j

        msd = []
        for dt in range(msd_matrix.shape[0]):
            msd.append(np.mean(msd_matrix[dt]))

        msd = np.array(msd)
        # msd /= self.diameter
        time = np.linspace(0.,float(self.nframes) / float(self.frame_rate),self.nframes)
        if self.plot == True:
            pl.figure()
            pl.loglog(time[:self.points],msd[:self.points],lw = 0, marker = 'o',mfc = 'w',mec = 'b')

            if self.references == True:
                pl.plot(time,msd.max()*(time**2),c = 'k',alpha = 0.5,dashes = [6,3], label = '$t^{2}$')
                pl.plot(time,msd.max()*(time),c = 'k',alpha = 0.25,dashes = [4,2], label = 't')


            pl.xlabel('t / s')
            pl.ylabel('MSD')
            pl.ylim([1e-2, 1e4])


            if self.fit == True:
                if self.kind == 'ABP':
                    popt,pcov = curve_fit(self.ABP,time[:self.points],msd[:self.points],bounds=(0,[1.0,self.tau,self.velocity]))
                    pl.plot(time[:self.points],self.ABP(time[:self.points],*popt), c = 'r', label = 'ABP fit')
                    print('Dt = {}, tau = {}, v = {} '.format(popt[0],popt[1],popt[2]))
                if self.kind == 'Quincke':
                    popt,pcov = curve_fit(self.quincke,time[:self.points],msd[:self.points],bounds=(0,[self.tau,self.velocity]))
                    pl.plot(time[:self.points],self.quincke(time[:self.points],*popt), c = 'r', label = 'Quincke fit')
                    print('tau = {}, v = {}'.format(popt[0],popt[1]))
                if self.kind == 'Passive':
                    popt,pcov = curve_fit(self.passive,time[:self.points],msd[:self.points],bounds=(0,[1.]))
                    pl.plot(time[:self.points],self.passive(time[:self.points],*popt), c = 'r', label = 'Passive')
                pl.legend()
        else:
            pass

        if self.save == True:
            output = np.zeros((msd.shape[0],2))
            output[:,0] = time
            output[:,1] = msd
            np.savetxt(self.directory + 'data/msd.txt', output,fmt='%1.5f')

        return time, msd, popt
