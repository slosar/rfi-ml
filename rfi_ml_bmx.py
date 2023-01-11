import numpy as np
import os, datetime, pickle, glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

if not torch.cuda.is_available():
    print ("Warning: I see no CUDA, this will be slow!")
    
class BMXLoader:
    def __init__(self, Np=2048, freq='0000'):
        self.Np = Np
        self.freq = freq
        
    def loadData(self, data_dir, test=False):
        #print('Data dir:', data_dir+'/**/*_'+self.freq+'.npy')
        #data_files = glob.glob(data_dir+'/**/*_'+self.freq+'.npy')
        print('Data dir:', data_dir+'/**/*_'+self.freq+'.npy')
        data_files = glob.glob(data_dir+'/**/*_'+self.freq+'.npy')
        
        if test==False:
            data_0 = np.load(data_files[0])
            n_timestreams = np.size(data_0, 0)
            data_array = np.zeros((len(data_files)*n_timestreams, self.Np))
            for i, file in enumerate(data_files):
                data_file = np.load(file)
                for j in range(n_timestreams):
                    data_array[i*n_timestreams+j,:] += data_file[j]
        else:
        #Temporary for testing
            data_0 = np.load(data_files[0])
            n_timestreams = np.size(data_0, 0)
            data_array = np.zeros((10, self.Np))
            data_file = np.load(data_files[0])
            for j in range(10):
                data_array[j,:] += data_file[j]
        
        print('Data Loaded!')
        print('Size: ',np.shape(data_array))
        return data_array
    
    def normalizeData(self, train_data, eval_data):
        #Normalize mean to zero
        train_data -= np.mean(train_data)
        eval_data -= np.mean(eval_data)
        print('Train data mean: ',np.mean(train_data))
        print('Eval data mean: ',np.mean(eval_data))
        
        #Normalize RMS to RMS of first timestream
        rms_norm = np.std(train_data[0])
        print("RMS normalization factor: ",rms_norm)
        #Or normalize to RMS ~0.0175, approximate value from test version?
        #rms_norm = 0.0175
        for i in range(np.size(train_data, 0)):
            rms = np.sqrt(np.sum((train_data[i])**2)/self.Np)
            train_data[i] *= (rms_norm/rms)
        for i in range(np.size(eval_data, 0)):           
            rms = np.sqrt(np.sum((eval_data[i])**2)/self.Np)
            eval_data[i] *= (rms_norm/rms)
            
        return train_data, eval_data
            

class ToyGenerator:
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

    def getGaussian(self):
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

    def getNonGaussianLocalized(self, freq=(0.2, 0.5), sigma=(20, 50), ampl=(0.1, 0.2)):
        """ Returns a certain type of localized non-Gaussian signal """
        # Signal with non-Gaussian shape
        freq = np.random.uniform(*freq)
        phase = np.random.uniform(0, 2 * np.pi)
        sigma = np.random.uniform(*sigma)
        #sigma = sigma[0]
        pos = np.random.uniform(3 * sigma, self.Np - 3 * sigma)
        ampl = np.random.uniform(*ampl)
        rfi = (
            ampl
            * np.cos(phase + freq * self.t)
            * np.exp(-(self.t - pos) ** 2 / (2 * sigma ** 2))
        )        
        return rfi

    def getNonGaussianNonlocalized(self, freq=(0.2, 0.5), ampl=(0.4, 0.8), Pflip=0.5):
        """ Returns a certain type of nonlocalized non-Gaussian signal """
        freq = np.random.uniform(*freq)
        phase = np.random.uniform(0, 2 * np.pi)
        ampl = np.random.uniform(*ampl)
        if np.random.uniform(0, 1) < Pflip: #flip sign of cosine at random point in timestream, with given probability
            flip = self.Np * np.random.uniform(0, 1) 
        else:
            flip = self.Np
        rfi = (ampl * np.cos(phase + freq * self.t) * np.where(self.t<=flip, 1, -1))      
        return rfi
    
class RFIDetect:

    # Decoder nework
    class Decoder(nn.Module):
        def __init__(self, z_dim, hidden_dim, out_dim):
            super(RFIDetect.Decoder, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.02, inplace=False),
                #nn.Linear(hidden_dim, hidden_dim),
                #nn.LeakyReLU(0.02, inplace=False),
                #nn.Linear(hidden_dim, hidden_dim),
                #nn.LeakyReLU(0.02, inplace=False),
                #nn.Linear(hidden_dim * 16, hidden_dim * 32),
                #nn.LeakyReLU(0.02, inplace=False),
                #nn.Linear(hidden_dim * 32, hidden_dim * 64),
                #nn.LeakyReLU(0.02, inplace=False),
                #nn.Linear(hidden_dim * 64, hidden_dim * 128),
                #nn.LeakyReLU(0.02, inplace=False),
                nn.Linear(hidden_dim, out_dim, bias=False),
            )

        def forward(self, x):
            out = self.main(x)
            return out

    # Encoder network
    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, z_dim):
            super(RFIDetect.Encoder, self).__init__()

            self.main = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                #nn.LeakyReLU(0.02, inplace=False),
                #nn.Linear(hidden_dim * 128, hidden_dim * 64),
                #nn.LeakyReLU(0.02, inplace=False),
                #nn.Linear(hidden_dim * 64, hidden_dim * 32),
                #nn.LeakyReLU(0.02, inplace=False),
                #nn.Linear(hidden_dim * 32, hidden_dim * 16),
                #nn.LeakyReLU(0.02, inplace=False),
                #nn.Linear(hidden_dim, hidden_dim),
                #nn.LeakyReLU(0.02, inplace=False),
                #nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, z_dim),
            )

        def forward(self, x):
            out = self.main(x)
            return out
        
    def __init__(self, Np, z_dim = 16, hidden_dim = 256, nworkers = 0, Nepochs = 3):
        self.Np = Np
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.nworkers = nworkers
        self.Nepochs = Nepochs
        
        self.wrapper = 'example.ipynb'
        self.code = 'rfi_ml.py'
        self.save_folder = 'rfi_ml/'
        os.makedirs(self.save_folder, exist_ok=True)
        self.save_time = str(datetime.datetime.now()).split('.')[0].replace(' ','_').replace(':','-')
        
        
        os.system('scp ./' + self.wrapper + ' ' + self.save_folder + '/' + self.save_time + '_' + self.wrapper)
        os.system('scp ../' + self.code + ' ' + self.save_folder + '/' + self.save_time + '_' + self.code)

    def Gaussianize(self, signal):
        """ Gaussianizes a signal """
        fsig = np.fft.rfft(signal)
        rot = np.exp(1j * np.random.uniform(0, 2 * np.pi, len(fsig)))
        return np.fft.irfft((fsig * rot))

    def train(self, train_array, gauss_fact=torch.ones(1), lamb=1, batch_size = 32, lr=0.0002, betas=(0.5, 0.999)):
        train_tensor = torch.from_numpy(train_array)
        s_trainloader = DataLoader(
            torch.utils.data.TensorDataset(train_tensor),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.nworkers,
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
        #recons_criterion = nn.L1Loss()
        
        iters = 0 
        
        #Training loop
        for epoch in range(self.Nepochs):
            # iterate through the dataloaders
            #for i, (g, ng) in enumerate(zip(g_trainloader, ng_trainloader)): 
            for i, s in enumerate(s_trainloader):
                # set to train mode
                self.netE.train()
                self.netD.train()
                             
                #g = g[0].float().cuda()
                #ng = ng[0].float().cuda()
                #s = g + ng
                s = s[0].float().cuda()
                
                #sigs = g + ng
                #gaussianized = torch.stack([gauss_fact * self.Gaussianize(sig.cpu()) for sig in sigs]).cuda().float()
                #modsig = sigs + gaussianized
                #modsig = np.sqrt(1-lamb**2)*sigs + lamb*gaussianized #Normalization option
                #modsig = g + ng
                
                # encode-decode
                #recons_out = self.netD(self.netE(modsig))
                #recons_out = self.netD(self.netE(g + ng)) #Reducing number of variables to reduce memory load
                recons_out = self.netD(self.netE(s))
                
                # loss
                #loss = recons_criterion(modsig - recons_out, lamb*gaussianized) #Normalization option
                #loss = recons_criterion(g + ng - recons_out, torch.zeros(recons_out.size()).float().cuda())
                #loss = recons_criterion(g + ng, recons_out) #Reduced number of variables
                loss = recons_criterion(s, recons_out)
                
                # backpropagate and update the weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print the training losses
                if iters % 100 == 0:
                    print(
                    "[%3d/%d][%3d/%d]\tLoss: %.10f"
                    % (epoch, self.Nepochs, i, len(s_trainloader), loss.item())
                    )
                iters += 1
        
    def evaluate(self, test_array, gauss_fact=torch.ones(1), lamb=1):
            
        self.netE.eval()
        self.netD.eval()
        
        recons_out = []       
        with torch.no_grad():
            #sigs = test_array.float().cuda()
            #gaussianized = torch.stack([gauss_fact * torch.from_numpy(self.Gaussianize(sig.cpu().numpy())) for sig in sigs]).cuda().float()
            #modsig = sigs + gaussianized
            #modsig = np.sqrt(1-lamb**2)*sigs + lamb*gaussianized
            modsig = torch.from_numpy(test_array).float().cuda()
            recons_out = self.netD(self.netE(modsig))
          
        self.rms = np.std(modsig.cpu().numpy(), axis=1)
        self.avg_rms = np.mean(self.rms)
        #self.rms = np.sqrt(np.sum((recons_out.cpu().numpy()-ng_test_array.cpu().numpy())**2, axis=1)/self.Np)
        #self.avg_rms = np.sum(self.rms)/ng_test_array.cpu().numpy().shape[0]
        #self.rfi_pwr_remaining = np.sum((ng_test_array.cpu().numpy()-recons_out.cpu().numpy())**2, axis=1)
        #self.frac_rfi_pwr = self.rfi_pwr_remaining/np.sum(ng_test_array.cpu().numpy()**2, axis=1)
        #self.frac_sig_pwr = self.rfi_pwr_remaining/np.sum(g_test_array.cpu().numpy()**2, axis=1)
        #self.avg_rfi_pwr_remain = np.sum(self.frac_rfi_pwr)/ng_test_array.cpu().numpy().shape[0]
        #self.avg_gauss = np.sum(np.abs(g_test_array.cpu().numpy()),axis=None)/(self.Np*g_test_array.cpu().numpy().shape[0])

        print("Epochs: ",self.Nepochs)
        print("N tests: ", test_array.shape[0])
        print("Length of timestream: ", self.Np)
        print("Input Timestream RMS: ", self.rms)
        print("Avg RMS for all tests: ", self.avg_rms, "\n") 
        #print("Timestream Obs-Expect RMS: ", self.rms)
        #print("Avg RMS for all tests: ", self.avg_rms, "\n")    
        #print("Fraction of RFI power remaining: ", self.frac_rfi_pwr)
        #print("Remainder as fraction of signal power: ", self.frac_sig_pwr)
        #print("Avg of RFI power remaining: ",self.avg_rfi_pwr_remain)
        #print("Avg amp of Gaussian signal: ",self.avg_gauss)
        
        save_filename = self.save_time + '_epoch_' + str(self.Nepochs).zfill(7) + '_eval_data'
        save_path = os.path.join(self.save_folder, save_filename)
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)

        return recons_out
    
    def plot_eval(self, recons_out, test_array):
        #Test diagnostics at end of epoch        
        time = range(self.Np)

        for test_int in range(len(test_array)):
            
            """
            #Plot components separately
            fig = plt.figure(figsize=(17,10)) 
            ax = fig.add_subplot(1,2,1)
            ax.plot(time, test_array[test_int]) # ng
            ax.set_title("Input Timestream")

            ax = fig.add_subplot(1,2,2)
            ax.plot(time, recons_out[test_int,:].cpu().numpy()) # output
            ax.set_title("Network Output (NG)")

            save_filename = self.save_time + '_epoch_' + str(self.Nepochs).zfill(7) + '_test_' + str(test_int) + '.pdf'
            save_path = os.path.join(self.save_folder, save_filename)
            print('Saving file...{}'.format(save_path))
            fig.savefig(save_path, bbox_inches='tight')
            
            fig.clf()
            plt.close(fig)
            """
            
            #Overplot
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(test_array[test_int,:])
            ax.plot(recons_out[test_int,:].cpu().numpy())
            ax.legend(['Timestream In','RFI Recovered'])
    
            save_filename = self.save_time + '_overplot_test_' + str(test_int) + '.png'
            save_path = os.path.join(self.save_folder, save_filename)
            print('Saving file...{}'.format(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            
            fig.clf()
            plt.close(fig)
            
        #rand int seed check
        #print('Numpy rand int check: ',np.random.uniform(0,1,1))
        #print('Torch rand int check: ',torch.rand(1))