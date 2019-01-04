import numpy as np
import pandas as pd
import scipy as sp
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import kurtosis, skew
from scipy import fftpack
from sklearn import preprocessing
from sklearn.cluster import KMeans

 
''' FFT analysis on each data streams for classifiction '''
''' Statistical analysis of FFT data: Here we just consider only one frame of data which consists of current timeslot data and previous X timeslots data both in uplink and downlink '''        
        
class Stat_FFT():    
    def __init__(self,Tc,Tp,NSTA,UL,DL,Fs,N,N_feat=11):
        print(' Statistical analysis of FFT data ');
        self.Tc = Tc # Number of time slots for current frame 
        self.Tp = Tp # Number of time slots for previous frame
        self.N_STA = NSTA
        self.UL = UL
        self.DL = DL
        self.N_feat = N_feat; # number of features per station
        self.N = N; # number of points to consider for FFT;
        self.Fs = Fs; # sampling frequency for FFT
        
        ''' current Tc data frames '''
        self.DL_Tc  =  np.zeros((self.N_STA,self.Tc))
        self.UL_Tc  = np.zeros((self.N_STA,self.Tc))

        ''' Previous Tp data frames '''
        self.DL_Tp  =  np.zeros((self.N_STA,self.Tp))
        self.UL_Tp  = np.zeros((self.N_STA,self.Tp))

        
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
    def calc_stats(self,Ti):
        ''' time shift by step 1 for the next time slot data '''
        self.DL_Tc = np.roll(self.DL_Tc,-1,axis=1)
        self.UL_Tc = np.roll(self.UL_Tc,-1,axis=1)# left shift the data 

        self.DL_Tp = np.roll(self.DL_Tp,-1,axis=1)
        self.UL_Tp = np.roll(self.UL_Tp,-1,axis=1)# left shift the data 
        
        STA_stat =  np.zeros((self.N_STA,self.N_feat))
        
        for j in range(self.N_STA):
            ''' form frame of length Tc and Tp for current and previous respectively'''
            self.DL_Tc[j,-1] =  self.DL[j,Ti] 
            self.UL_Tc[j,-1] =  self.UL[j,Ti]
            
            # collect features for station j
            STA_j = self.stat_anal(self.DL_Tc[j]);
            STA_j.extend(self.stat_anal(self.UL_Tc[j]));
            #STA_j.extend([self.ul_to_dl(self.UL[j,Ti],self.DL[j,Ti])])
            STA_stat[j,:] = np.array(STA_j);#STA_j;            
        STA_stat[np.isnan(STA_stat)] = 0; # check for NaN values and make it 0   
        return STA_stat
    
    
'''================================================================================================='''

class STA_clustering():
    def __init__(self,Tc,Tp,N_feat,Fs,DL_R,UL_R,N_STA):
        self.Tc=Tc; # current time slot
        self.Tp=Tp; # previous time slot
        self.N_feat =  N_feat; # number of features for clustering
        self.Fs = Fs; #Sampling frequency
        self.DL_R = DL_R; # Downlink bitrates of STAs
        self.UL_R = UL_R; # Uplink bitrates of STAs
        self.N_STA = N_STA; # Num of STAs in the network
        self.FFT_anal = Stat_FFT(Tc,Tp,N_STA,UL_R,DL_R,Fs,Tc,N_feat)
    
    def get_cluster(self,Ti,km):
        STA_stats = self.FFT_anal.calc_stats(Ti)  # get STA Stats from Fourier analysis
        STA_stat_norm = preprocessing.scale(STA_stats) # normalize data to avoid  
                                                       #biasing effect
        #print(STA_stat_norm)
        if Ti<20:
            km = KMeans(n_clusters=10,init='k-means++',n_init=10,verbose=0,random_state=0).fit(STA_stat_norm) 

        #         labels = km.labels_
        labels = km.predict(STA_stat_norm)
#         #STAs_Tc = np.transpose(np.array([DL_Tc[:,-1],UL_Tc[:,-1],labels]))
        plot_clustering = True
        if plot_clustering:
            plt.figure(Ti)
            plt.hist(labels)
            plt.grid()
        return labels,km
        
'''================================================================================================='''
'''
 This class is basically used for implementing station allocation algorithms such as: 

1) Distance based Station allocation to APs

2) Random allocation of STAs to APs

3) equal number of STAs allocation to APs 

4) Monte carlo- some optimized allocation of APs

'''

''' Station Mapping Algorithms '''    
class Station_map():
    def __init__(self,area,nAPs,nSTAs,loc_AP):
        self.nAPs = nAPs;
        self.nSTAs = nSTAs;
        self.loc =  loc_AP;
        self.loc =  self.loc[0:nAPs];
        self.area = area;

    def dist(self,sta_pos):
        return np.sum((self.loc-sta_pos)**2,axis=1);

    ''' Distance based STA allocation to APs '''
    def dist_based(self,sta_pos):
        STA_map = np.zeros(shape = [self.nSTAs])
        for i in range(self.nSTAs):
            STA_map[i] = np.argmin(self.dist(sta_pos[i]))
        return STA_map

    ''' Random assignment of STAs to APS '''
    def rand(self):
        ''' return random station map '''
        return np.random.randint(self.nAPs,size=self.nSTAs)

    ''' equal number of STAs to APs '''
    def equal(self,best,current,STA_map_p):
        if current>best:
            STA_map=STA_map_p;
            best=current;
        else:
            STA_map = [];
            for i in range(self.nAPs):
                STA_map.extend([i]*int(self.nSTAs/self.nAPs))
            STA_map = np.array(STA_map)
            np.random.shuffle(STA_map) # shuffle the positions of the STAs  to AP mapping
            best=best;
        return STA_map,best;
    
    ''' Decision Tree based mapping '''
    def dec_tree(self,Dr,Ur,sta_pos):
        STA_map = np.zeros(shape = [self.nSTAs])
        for i in range(self.nSTAs):
            if Dr[i]>150e3 and Ur[i]>150e3 and np.argmin(self.dist(sta_pos[i]))<=5:
                STA_map[i] = 0;
            else:
                STA_map[i] = 1;
        return STA_map
    
    ''' monte-carlo station allocation '''
    def mon_carlo(self,best,current,STA_map_p):
        if current>best:
            STA_map = STA_map_p;
            best=current;
        else:
            STA_map = self.rand()
            best=current;
        return STA_map,best;
    
    

        
        
        
    
    
    
        
        

