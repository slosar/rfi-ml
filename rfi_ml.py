import numpy as np
import os, datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

if not torch.cuda.is_available():
    print ("Warning: I see no CUDA, this will be slow!")

class ToyGenerator:
    def __init__(self, N=1024, Pk=None):
        self.N = N
        self.Nfft = self.N // 2 + 1
        self.k = np.linspace(0, 1, self.Nfft)
        self.t = np.linspace(0, 1, self.N)

        if Pk is None:
            self.Pk = (1 + np.exp(-(self.k - 0.5) ** 2 / (2 * 0.1 ** 2))) * np.exp(-self.k / 0.5)
        else:
            self.Pk = Pk

    def getGaussian(self):
        """ Returns Gaussian signal with a known power spectrum """
        xf = np.random.normal(0.0, 1.0, self.Nfft) + 1j * np.random.normal(0.0, 1.0, self.Nfft)
        xf *= self.Pk
        return np.fft.irfft(xf)

    def getNonGaussianLocalized(
        self, freq=(200, 500), sigma=(0.02, 0.05), ampl=(0.1, 0.2)
    ):
        """ Returns a certain type of non-Gaussian signal """
        # Signal with non-Gaussian shape
        freq = np.random.uniform(*freq)
        phase = np.random.uniform(0, 2 * np.pi)
        #sigma = np.random.uniform(*sigma)
        sigma = sigma[0]
        pos = np.random.uniform(3 * sigma, 1 - 3 * sigma)
        ampl = np.random.uniform(*ampl)
        rfi = (
            ampl
            * np.cos(phase + freq * self.t)
            * np.exp(-(self.t - pos) ** 2 / (2 * sigma ** 2))
        )
        #print('freq: ', freq)
        #print('phase: ', phase)
        #print('pos: ', pos)
        #print('ampl: ',ampl)
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

    def __init__(self, Np, z_dim = 16, hidden_dim = 256, ncores = 4):
        self.Np = Np
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.ncores = ncores

    def Gaussianize(self, signal):
        """ Gaussianizes a signal """
        fsig = np.fft.rfft(signal)
        rot = np.exp(1j * np.random.uniform(0, 2 * np.pi, len(fsig)))
        return np.fft.irfft((fsig * rot))

    def train(self, g_train_array, ng_train_array, gauss_fact=torch.ones(1), batch_size = 32, epochs = 3, lr=0.0002, betas=(0.5, 0.999)):

        g_trainloader = DataLoader(
            torch.utils.data.TensorDataset(g_train_array),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.ncores,
            pin_memory=True,
            drop_last=True,
        )

        ng_trainloader = DataLoader(
            torch.utils.data.TensorDataset(ng_train_array),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.ncores,
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

        #Training criterion
        recons_criterion = nn.MSELoss()
        
        iters = 0 
        
        #Training loop
        for epoch in range(epochs):
            # iterate through the dataloaders
            for i, (g, ng) in enumerate(zip(g_trainloader, ng_trainloader)):             
                # set to train mode
                self.netE.train()
                self.netD.train()
                             
                g = g[0].float().cuda()
                ng = ng[0].float().cuda()
                
                sigs = g + ng
                gaussianized = torch.stack([gauss_fact * self.Gaussianize(sig.cpu()) for sig in sigs]).cuda().float() #"gaussianized"
                modsig = sigs + gaussianized
                
                # encode-decode
                recons_out = self.netD(self.netE(modsig))
                # loss
                loss = recons_criterion(modsig - recons_out, gaussianized)
                # backpropagate and update the weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print the training losses
                if iters % 100 == 0:
                    print(
                    "[%3d/%d][%3d/%d]\tLoss: %.10f"
                    % (epoch, epochs, i, len(g_trainloader), loss.item())
                    )
                iters += 1
            #self.netE.eval()
            #self.netD.eval()
        
    def evaluate(self, g_test_array, ng_test_array, gauss_fact=torch.ones(1)):
        
        #if not hasattr(self,"netD"):
        #    self.netD = self.Decoder(
        #        z_dim=self.z_dim, hidden_dim=self.hidden_dim, out_dim=self.Np
        #    ).cuda()
        #    self.netE = self.Encoder(
        #        input_dim=self.Np, hidden_dim=self.hidden_dim, z_dim=self.z_dim
        #    ).cuda()
            
        self.netE.eval()
        self.netD.eval()
        
        recons_out = []
        #test = torch.stack([tin for tin in problem])
        #testloader = DataLoader(
        #    torch.utils.data.TensorDataset(test),
        #    batch_size=len(test),
        #    shuffle=True,
        #    num_workers=self.ncores,
        #    pin_memory=True,
        #    drop_last=True,
        #)
        
        with torch.no_grad():
            sigs = g_test_array.float().cuda() + ng_test_array.float().cuda()
            gaussianized = torch.stack([gauss_fact * torch.from_numpy(self.Gaussianize(sig.cpu().numpy())) for sig in sigs]).cuda().float()
            modsig = sigs + gaussianized
            recons_out = self.netD(self.netE(modsig))
            #for line in testloader:
            #    line = line[0].float().cuda()
            #    recons_out = self.netD(self.netE(line))
            #    output.append(recons_out.cpu().numpy())

            print(recons_out)
        return recons_out
    
    def plot_eval(self, recons_out, g_test_array, ng_test_array, Nepochs):
        #Test diagnostics at end of epoch
        save_folder = 'rfi_ml/'
        os.makedirs(save_folder, exist_ok=True)
        save_time = str(datetime.datetime.now()).split('.')[0].replace(' ','_').replace(':','-')
        
        time = range(self.Np)
        sig = (g_test_array.cpu() + ng_test_array.cpu()).numpy()

        for test_int in range(len(g_test_array)):
            gauss = self.Gaussianize(sig[test_int,:])
            
            #Plot components separately
            plt.figure(figsize=(17,10)) 
            ax = plt.subplot(2,3,1)
            plt.plot(time, ng_test_array[test_int].cpu().numpy()) # ng
            ax.set_title("ng")

            ax = plt.subplot(2,3,2)
            plt.plot(time, g_test_array[test_int].cpu().numpy()) # g
            ax.set_title("g")

            ax = plt.subplot(2,3,3)
            plt.plot(time, sig[test_int,:]) # g + ng 
            ax.set_title("sig = ng + g")

            ax = plt.subplot(2,3,4)
            plt.plot(time, gauss) # gaussianize(ng + g)
            ax.set_title("Gauss(sig)")

            ax = plt.subplot(2,3,5)
            plt.plot(time, sig[test_int,:]+gauss) # input
            ax.set_title("network input=Gauss(sig) + sig")

            ax = plt.subplot(2,3,6)
            plt.plot(time, recons_out[test_int,:].cpu()) # output
            ax.set_title("network output (~ng)")

            save_filename = save_time + '_epoch_' + str(Nepochs).zfill(7) + '_test_' + str(test_int) + '.png'
            save_path = os.path.join(save_folder, save_filename)
            print('Saving file...{}'.format(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            #Overplot
            plt.plot(sig[test_int,:])
            plt.plot(recons_out[test_int,:].cpu())
            plt.plot(ng_test_array[test_int])
            plt.legend(['Test In','Recovered','RFI'])
    
            save_filename = save_time + '_overplot_test_' + str(test_int) + '.png'
            save_path = os.path.join(save_folder, save_filename)
            print('Saving file...{}'.format(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
        #rand int seed check
        print('Numpy rand int check: ',np.random.uniform(0,1,1))
        print('Torch rand int check: ',torch.rand(1))