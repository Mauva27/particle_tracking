import numpy as np
from scipy.optimize import curve_fit
from ..utils import get_trajectories_list
import pylab as pl
import ipdb


class Qt:
    def __init__(self,input,directory,data,params,plot,fit,points,tau,save):
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
            self.data = np.loadtxt(self.directory + 'data/filtered_coords.txt')
            self.nframes = self.data[:,2].max() + 1
        self.frame_rate = params['frame_rate']
        self.plot = plot
        self.fit = fit
        self.points = points
        self.tau = tau
        self.save = save

    def se_fit(self,t,tau,b,c):
        return c*np.exp(-(t/tau)**b)

    def ex_fit(self,t,c,t0,tau,b):
        return c*np.exp(-t/t0) + (1-c)*np.exp(-(t/tau)**b)

    def compute_qt(self,traj):
        coords = traj[:,:2]
        dr2 = np.array([np.linalg.norm(coords[dt] - coords[0]) for dt in range(coords.shape[0])])**2
        return np.exp(-dr2/(self.diameter**2))

    def mean_qt(self):
        qt = np.zeros(self.nframes)

        if type(self.data) ==  dict:
            trajs = sorted(self.data.keys())
            for t in trajs:
                t = int(t)
                qt[:int(self.data[t].shape[0])] += self.compute_qt(self.data[t])
            qt /= len(trajs)
        if type(self.data) == np.ndarray:
            trajs = get_trajectories(self.data)
            for traj in trajs:
                qt[:int(traj.shape[0])] += self.compute_qt(traj)
            qt /= len(trajs)

        time= np.arange(1,len(qt)+1) / self.frame_rate
        popt, pcov = curve_fit(self.se_fit,time[:self.points],qt[:self.points], bounds = (0,[self.tau,1.0,1.0]))
        popt_, pcov_ = curve_fit(self.ex_fit,time[:self.points],qt[:self.points], bounds = (0,[1.0,self.tau,self.tau,1.0]))


        if self.plot == True:
            pl.figure()
            pl.semilogx(time,qt,lw=0, marker = 'o', mfc = 'None',mec = 'b',alpha = 0.75)
            if self.fit == True:
                pl.plot(time[:self.points],self.se_fit(time[:self.points],*popt), c = 'r', label = 'SE')
                pl.plot(time[:self.points],self.ex_fit(time[:self.points],*popt_), c = 'g', label = 'Other')
                print ('tau = {}, b = {}, c = {}'.format(popt[0],popt[1],popt[2]))
                print ('Other tau = {}, b = {}, c = {}'.format(popt_[2],popt_[3],popt_[0]))

                pl.legend()


            pl.xlabel('t / s')
            pl.ylabel('Q(t)')
            pl.ylim(0,1.1)

        if self.save == True:
            output = np.zeros((qt.shape[0],2))
            output[:,0] = time
            output[:,1] = qt
            np.savetxt(self.directory + 'data/qt.txt', output,fmt='%1.5f')

            output1 = np.zeros((1,popt.shape[0]))
            output1[:,0] = popt[0]
            output1[:,1] = popt[1]
            output1[:,2] = popt[2]
            np.savetxt(self.directory + 'data/qt_fit_params.txt',output1,fmt='%1.5f',header = 'tau b c',comments = '')

        return time,qt,popt
