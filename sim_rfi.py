import numpy as np
import os, datetime, pickle, glob

import torch.optim as optim
import matplotlib.pyplot as plt

if not torch.cuda.is_available():
    print ("Warning: I see no CUDA, this will be slow!")

class RFI:
    def __init__(self, Np=1024, Pk=None):
        self.Np = Np
        self.Nfft = self.Np // 2 + 1
        self.k = np.linspace(0, self.Nfft, self.Nfft)
        self.t = np.linspace(0, self.Np, self.Np)

        if Pk is None:
            self.Pk = (1 + np.exp(-(self.k - 256) ** 2 / (2 * 50 ** 2))) * np.exp(-self.k / 256)
            #self.Pk = (1 + np.exp(-(self.k - 0.5) ** 2 / (2 * 0.1 ** 2))) * np.exp(-self.k / 0.5)
        else:
            self.Pk = Pk
            
        #ax = plt.subplot(1,1,1)
        #plt.plot(self.k, self.Pk) #gaussian power spectrum
        #ax.set_title("Gaussian Signal Power Spectrum")

    def Gauss(self):
        """ Returns Gaussian signal with a known power spectrum """
        xf = np.random.normal(0.0, 1.0, self.Nfft) + 1j * np.random.normal(0.0, 1.0, self.Nfft)
        xf *= self.Pk        
        xf /= 4.0*np.sum(np.abs(xf**2)) #Normalize for varying timestream length

        #print("Avg Pk: ", np.mean(self.Pk))
        #print("Avg xf: ", np.abs(np.mean(xf)))
        #print("Avg irfft(xf): ", np.abs(np.mean(np.fft.irfft(xf, norm="forward"))))
        #print("Avg2 irfft(xf): ", np.abs(np.mean(np.fft.irfft(xf)*self.Np)))

        #return np.fft.irfft(xf, norm="forward") #Forward keywork for fft normalization prevents divide by 1/Nfft on irfft
        return np.fft.irfft(xf)*self.Np #Manual canceling of the normalization for backwards compatibility with numpy <v1.20

    def linear

    def leakyrelu

    def exp

    def damped

    def ngl(self, freq=(0.2, 0.5), sigma=(20, 50), ampl=(0.1, 0.2)):
        """ Returns a certain type of localized non-Gaussian signal """
        # Signal with non-Gaussian shape
        freq = np.random.uniform(*freq)
        phase = np.random.uniform(0, 2 * np.pi)
        sigma = np.random.uniform(*sigma)
        pos = np.random.uniform(3 * sigma, self.Np - 3 * sigma)
        ampl = np.random.uniform(*ampl)
        rfi = (
            ampl
            * np.cos(phase + freq * self.t)
            * np.exp(-(self.t - pos) ** 2 / (2 * sigma ** 2)) 
        )    
        rfi_pwr = np.sum((rfi[round(pos-3*sigma):round(pos+3*sigma)])**2)/(6*sigma)
        return rfi, rfi_pwr

    def ngnl(self, freq=(0.2, 0.5), ampl=(0.05, 0.05), Pflip=1.0):
        """ Returns a certain type of nonlocalized non-Gaussian signal """
        freq = np.random.uniform(*freq)
        phase = np.random.uniform(0, 2 * np.pi)
        ampl = np.random.uniform(*ampl)
        if np.random.uniform(0, 1) < Pflip: #flip sign of cosine at random point in timestream, with given probability
            flip = self.Np * np.random.uniform(0, 1) 
        else:
            flip = self.Np
        rfi = (ampl * np.cos(phase + freq * self.t) * np.where(self.t<=flip, 1, -1))  
        rfi_pwr = np.sum(rfi**2)/(self.Np)
        return rfi, rfi_pwr

    def burst
    
