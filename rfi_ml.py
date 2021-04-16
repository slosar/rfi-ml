import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

if not torch.cuda.is_available():
    print ("Warning: I see no CUDA, this will be slow!")

class ToyGenerator:
    def __init__(self, N=1024, Pk=None):
        self.N = N
        self.Nfft = self.N // 2 + 1
        self.k = np.linspace(0, 1, self.Nfft)
        self.t = np.linspace(0, 1, self.N)

        if Pk is None:
            self.Pk = (1 + np.exp(-(self.k - 0.5) ** 2 / (2 * 0.1 ** 2))) * np.exp(
                -self.k / 0.5
            )
        else:
            self.Pk = Pk

    def getGaussian(self):
        """ Returns Gaussian signal with a known power spectrum """
        xf = np.random.normal(0, 1.0, self.Nfft) + 1j * np.random.normal(
            0.0, 1.0, self.Nfft
        )
        xf *= self.Pk
        return np.fft.irfft(xf)

    def getNonGaussianLocalized(
        self, freq=(200, 500), sigma=(0.02, 0.05), ampl=(0.1, 0.2)
    ):
        """ Returns a certain type of non-Gaussian signal """
        # Signal with non-Gaussian shape
        freq = np.random.uniform(*freq)
        phase = np.random.uniform(0, 2 * np.pi)
        sigma = np.random.uniform(*sigma)
        pos = np.random.uniform(3 * sigma, 1 - 3 * sigma)
        ampl = np.random.uniform(*ampl)
        rfi = (
            ampl
            * np.cos(phase + freq * self.t)
            * np.exp(-(self.t - pos) ** 2 / (2 * sigma ** 2))
        )
        return rfi


class RFIDetect:

    # Decoder nework
    class Decoder(nn.Module):
        def __init__(self, z_dim, hidden_dim, out_dim):
            super(RFIDetect.Decoder, self).__init__()
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
            super(RFIDetect.Encoder, self).__init__()

            self.main = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, z_dim),
            )

        def forward(self, x):
            out = self.main(x)
            return out

    def __init__(self, Np, z_dim = 16, hidden_dim = 256 ):
        self.Np = Np
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

    def Gaussianize(self, signal):
        """ Gaussianizes a signal """
        fsig = np.fft.rfft(signal)
        rot = np.exp(1j * np.random.uniform(0, 2 * np.pi, len(fsig)))
        return np.fft.irfft((fsig * rot))

    def train(self, input_data, gauss_fact=1.0, batch_size = 32, epochs = 30, lr=0.0002, betas=(0.5, 0.999)):

        gaussianized = [gauss_fact * self.Gaussianize(s) for s in input_data]
        train_in = [np.array(i+g) for i,g in zip(input_data,gaussianized)]

        train_in = torch.stack([torch.from_numpy(tin) for tin in train_in])
        train_out = torch.stack([torch.from_numpy(tout) for tout in gaussianized])

        #train_in = torch.from_numpy(np.array(train_in))
        #train_out = torch.from_numpy(np.array(gaussianized))

        in_trainloader = DataLoader(
            torch.utils.data.TensorDataset(train_in),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        out_trainloader = DataLoader(
            torch.utils.data.TensorDataset(train_out),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        if not hasattr(self,"netD"):
            self.netD = self.Decoder(
                z_dim=self.z_dim, hidden_dim=self.hidden_dim, out_dim=self.Np
            ).cuda()
            self.netE = self.Encoder(
                input_dim=self.Np, hidden_dim=self.hidden_dim, z_dim=self.z_dim
            ).cuda()

        optimizer = optim.Adam(
            [{"params": self.netE.parameters()}, {"params": self.netD.parameters()}],
            lr=lr,
            betas=betas,
        )

        recons_criterion = nn.MSELoss()
        iters = 0 
        for epoch in range(epochs):
            # iterate through the dataloaders
            for i, (din, dout) in enumerate(zip(in_trainloader, out_trainloader)):
                # set to train mode
                self.netE.train()
                self.netD.train()
                din = din[0].float().cuda()
                dout = dout[0].float().cuda()
                # encode-decode
                recons_out = self.netD(self.netE(din))
                # loss
                loss = recons_criterion(din - recons_out, dout)
                # backpropagate and update the weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print the training losses
                if iters % 100 == 0:
                    print(
                    "[%3d/%d][%3d/%d]\tLoss: %.10f"
                    % (epoch, epochs, i, len(in_trainloader), loss.item())
                    )
                iters += 1
        self.netE.eval()
        self.netD.eval()
        
    def evaluate (self, problem):
        output = []
        test = torch.stack([torch.from_numpy(tin) for tin in problem])
        testloader = DataLoader(
            torch.utils.data.TensorDataset(test),
            batch_size=len(test),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        with torch.no_grad():
            for line in testloader:
                line = line[0].float().cuda()
                recons_out = self.netD(self.netE(line))
                output.append(recons_out.cpu().numpy())

        return output[0]
            
