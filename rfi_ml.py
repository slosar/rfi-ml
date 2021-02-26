import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


class ToyGenerator:
    def __init__ (self, N=1024, Pk=None):
        self.N = N
        self.Nfft = self.N // 2 + 1
        self.k = np.linspace(0,1,self.Nfft)
        self.t = np.linspace(0,1,self.N)

        if Pk is None:
            self.Pk = (1+np.exp(-(self.k-0.5)**2/(2*0.1**2)))*np.exp(-self.k/0.5)
        else:
            self.Pk = Pk

    def getGaussian(self):
        """ Returns Gaussian signal with a known power spectrum """
        xf = np.random.normal(0,1.,self.Nfft)+1j*np.random.normal(0.,1.,self.Nfft)
        xf *= self.Pk
        return np.fft.irfft(xf)

    def getNonGaussianLocalized(self, freq=(200,500), sigma=(0.02,0.05), ampl=(0.05,.02)):
        """ Returns a certain type of non-Gaussian signal """
        # Signal with non-Gaussian shape
        freq = np.random.uniform(*freq)
        phase = np.random.uniform(0,2*np.pi)
        sigma = np.random.uniform(*sigma)
        pos = np.random.uniform(3*sigma,1-3*sigma)
        ampl = np.random.uniform (*ampl)
        rfi = ampl*np.cos(phase+freq*self.t)*np.exp(-(self.t-pos)**2/(2*sigma**2))
        return rfi

    def Gaussianize(self, signal):
        """ Gaussianizes a signal """
        fsig = np.fft.rfft(signal)
        rot = np.exp(1j*np.random.uniform(0,2*np.pi,len(fsig)))
        return np.fft.irfft((fsig*rot))


# Decoder nework
class Decoder(nn.Module):
  def __init__(self, z_dim, hidden_dim, out_dim):
    super(Decoder, self).__init__()
    self.main = nn.Sequential(
      nn.Linear(z_dim, hidden_dim),
      nn.LeakyReLU(0.02, inplace=False),

      nn.Linear(hidden_dim, hidden_dim * 2),
      nn.LeakyReLU(0.02, inplace=False),

      nn.Linear(hidden_dim * 2, out_dim, bias=False),
    )
    
  def forward(self, x):
    out = self.main(x)
    return out

# Encoder network
class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, z_dim):
    super(Encoder, self).__init__()

    self.main = nn.Sequential(
      nn.Linear(input_dim, hidden_dim * 2),
      nn.LeakyReLU(0.02, inplace=False),

      nn.Linear(hidden_dim * 2, hidden_dim),
      nn.LeakyReLU(0.02, inplace=False),
      nn.Dropout(0.2),

      nn.Linear(hidden_dim, z_dim)
    )
    
  def forward(self, x):
    out = self.main(x)
    return out

