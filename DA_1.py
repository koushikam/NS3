import numpy as np
import pandas as pd
import scipy as sp
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import kurtosis, skew
from scipy import fftpack

''' =============================================================================================== '''

''' Extract UL data and DL data from each sample '''
class Get_data():
    def __init__(self):
        print('Gather Uplink and Downlink information ')
        
    ''' Break the data into the actual value based on the headers '''
    def get_each_values(self,file_data):
        [j_TS,j_DL,j_UL] = [0,2,1] # index values  
        [self.TS,self.DL_rate,self.UL_rate] = [[],[],[]] 
        for i in range(len(file_data)):
            self.TS.append(file_data[i][j_TS])
            self.DL_rate.append(file_data[i][j_DL])
            self.UL_rate.append(file_data[i][j_UL])
        return self.TS[:-1],self.DL_rate[:-1],self.UL_rate[:-1]
    
''' Python 3D plotting '''
def Plot_3D(x,y,z):
    # x is uplink and y is downlink 
    X,Y = np.meshgrid(x,y)
    Z = np.reshape(z,[len(x),len(y)])
    fig = plt.figure()    
    ax   = fig.gca(projection='3d')#fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(X,Y,np.transpose(Z),cmap=cm.jet,rstride=1, cstride=1, linewidth=0)
    ax.set_xlabel('No of STAs')
    ax.set_ylabel('# of time slots')
    ax.set_zlabel('Bitrate')
    ax.set_zlim(0, np.max(np.max(Z)))
    plt.grid()
    plt.show()
    
''' Auto correlation estimation '''
def autocorr(UL_rate,DL_rate,SHIFT):
    DL_ac = []
    UL_ac = []
    for shift in range(1,SHIFT):
        UL_shift=UL_rate[shift:]
        UL_nshift = UL_rate[:-shift]
        DL_shift=DL_rate[shift:]
        DL_nshift = DL_rate[:-shift]
        AC1 = np.corrcoef(UL_shift,UL_nshift)[1,0]
        UL_ac.append(AC1) 
        AC2 = np.corrcoef(DL_shift,DL_nshift)[0,1]
        DL_ac.append(AC2)        
    return UL_ac, DL_ac

''' =============================================================================================== '''

''' Statistical ananlysis on a time stamp basis '''

class Stat_Analysis():
    ''' initialize all the parameters used throughout the process using this class '''
    def __init__(self,Tc,Tp,NSTA,UL,DL):
        self.Tc = Tc # Number of time slots for current frame 
        self.Tp = Tp # Number of time slots for previous frame
        self.NSTA = NSTA
        self.UL = UL
        self.DL = DL
        
    ''' cross correlation '''    
    def cross_coff(self,Xc,Xp):
        x = np.corrcoef(Xc,Xp)[1,0]
        Xcor = 0 if np.isnan(x) else x
        return Xcor
    
    ''' Mean of the current frame '''
    def frame_mean(self,Xc):
        return np.mean(Xc)
        
    ''' Skewness of the current frame '''
    def frame_skew(self,Xc):
        SKEW = 0 if np.isnan(skew(Xc)) else (skew(Xc))
        return SKEW
    
    ''' Upload to Download ratio '''
    def ul_to_dl(self,ul,dl):
        Ratio = 0 if np.isnan((ul/dl)) or np.isinf((ul/dl)) else (ul/dl)
        return Ratio
    
    ''' perform fourier analysis of Tc point data rate vector '''
    def fft(self,X,f_s,plot):
        Y = fftpack.fft(X)
        if plot:
            freqs = fftpack.fftfreq(len(X)) * f_s # f_s corresponds to sampling frequency
            ''' plot fft magnitude plot '''
            fig, ax = plt.subplots()
            ax.stem(freqs, np.abs(Y),color='red')
            ax.set_xlabel('Frequency in Hertz [Hz]')
            ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
            ax.set_xlim(-f_s/2, f_s/2)
            plt.grid()
        return np.sum(np.abs(Y**2))/len(Y) # return energy associated with N-point fft
    
    ''' calculate over all the time duration  and generate future vectors for each station'''
    def calc_stats(self,Ti,DL_Tc,UL_Tc,DL_Tp,UL_Tp,N_feat,plot_fft=False):
        STA_stat =  np.zeros((self.NSTA,N_feat)) # each feature vector is of length 6 based on the statistical parameter estimationa nd download and upload bitrate information
        for j in range(self.NSTA):
            ''' form frame of length Tc and Tp for current and previous respectively'''
            DL_Tc[j,-1] =  self.DL[j,Ti] 
            UL_Tc[j,-1] =  self.UL[j,Ti]
            if Ti>self.Tp:
                DL_Tp[j][-1] =  self.DL[j,Ti-self.Tp]
                UL_Tp[j][-1] =  self.UL[j,Ti-self.Tp]
            
            # collect features for station j    
            STA_stat[j,0] = self.DL[j,Ti]
            STA_stat[j,1] = self.UL[j,Ti]
            STA_stat[j,2] = self.cross_coff(DL_Tc[j],DL_Tp[j])
            STA_stat[j,3] = self.frame_mean(DL_Tc[j])
            STA_stat[j,4] = self.frame_skew(DL_Tc[j])
            STA_stat[j,5] = self.fft(DL_Tc[j],1,plot_fft)
            STA_stat[j,6] = self.cross_coff(UL_Tc[j],UL_Tp[j])
            STA_stat[j,7] = self.frame_mean(UL_Tc[j])
            STA_stat[j,8] = self.frame_skew(UL_Tc[j])
            STA_stat[j,9] = self.ul_to_dl(self.UL[j,Ti],self.DL[j,Ti])
            STA_stat[j,10] = self.fft(UL_Tc[j],1,plot_fft)    
        return STA_stat,DL_Tc,UL_Tc,DL_Tp,UL_Tp
             
''' =============================================================================================== '''
        
''' Statistical analysis of FFT data: Here we just consider only one frame of data which consists of current timeslot data and previous X timeslots data both in uplink and downlink '''        
        
class Stat_FFT():
    
    def __init__(self,Tc,Tp,NSTA,UL,DL,Fs,N,N_feat=11):
        print(' Statistical analysis of FFT data ');
        self.Tc = Tc # Number of time slots for current frame 
        self.Tp = Tp # Number of time slots for previous frame
        self.NSTA = NSTA
        self.UL = UL
        self.DL = DL
        self.N_feat = N_feat; # number of features per station
        self.N = N; # number of points to consider for FFT;
        self.Fs = Fs; # sampling frequency for FFT
        
    def fft(self,X): # calculate fft for input X
        self.X = X;
        self.Y = fftpack.fft(X); # calculate FFT of the signal
        # frequency bins
        self.freq = fftpack.fftfreq(self.N) * self.Fs;
        ''' keep freq greater than 0 only since, negative frequency components are mirror image of 
        positive frequency values. In addition, we do not consider value at frequency bin zero since it 
        is associated closely to DC component and most of the time this term dominates over rest of the 
        frequency bins '''
        keep = (self.freq>0);
        self.Y = self.Y[keep];
        self.freq = self.freq[keep];
        
    ''' Upload to Download ratio '''
    def ul_to_dl(self,ul,dl):
        Ratio = 0 if np.isnan((ul/dl)) or np.isinf((ul/dl)) else (ul/dl)
        return Ratio
    
    ''' Evaluate some of the parameters based on the fft analysis ''' 
    def stat_anal(self,X):
        self.fft(X); # calculate fft
        SS =  np.sum(np.abs(self.Y)); ## calculating signal strength
        MS =  np.max(np.abs(self.Y))/SS; ## calculating maximum normalized amplitude
        FM =  np.argmax(np.abs(self.Y)); ## frequency of maximum amplitude
        ''' Power weighted average of all the frequencies, a center of gravity for the signal 
        frequencies''' 
        FC =  np.sum([self.freq[i]*np.abs(self.Y[i]) for i in range(len(self.Y))])/SS;
        FS =  np.sum([(self.freq[i]-FC)**2*np.abs(self.Y[i]) for i in 
                           range(len(self.Y))])/SS;        
        return [SS,MS,FM,FC,FS];
        
    ''' calculate over all the time duration  and generate future vectors for each station'''
    def calc_stats(self,Ti,DL_Tc,UL_Tc,DL_Tp,UL_Tp):
        STA_stat =  np.zeros((self.NSTA,self.N_feat))
        
        for j in range(self.NSTA):
            ''' form frame of length Tc and Tp for current and previous respectively'''
            DL_Tc[j,-1] =  self.DL[j,Ti] 
            UL_Tc[j,-1] =  self.UL[j,Ti]
            
            # collect features for station j
            STA_j = self.stat_anal(DL_Tc[j]);
            STA_j.extend(self.stat_anal(UL_Tc[j]));
            #STA_j.extend([self.ul_to_dl(self.UL[j,Ti],self.DL[j,Ti])])
            STA_stat[j,:] = np.array(STA_j);#STA_j;            
        STA_stat[np.isnan(STA_stat)] = 0; # check for NaN values and make it 0   
        return STA_stat,DL_Tc,UL_Tc,DL_Tp,UL_Tp
    
    
'''================================================================================================='''





















