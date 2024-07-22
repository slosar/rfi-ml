import numpy as np
import os, datetime, pickle, glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

if not torch.cuda.is_available():
    print ("Warning: I see no CUDA, this will be slow!")
    
class S4Loader:
    def __init__(self, Np=1024, freq='0000'):
        self.Np = Np
        self.freq = freq
        
    def loadData(self, data_dir, test=False):                     #this function may change based on how the data is organized
        print('Data dir:', data_dir+'/**/*_'+self.freq+'.npy')
        data_files = glob.glob(data_dir+'/**/*_'+self.freq+'.npy')
        
        if test==False:     #what does this mean?
            data_0 = np.load(data_files[0])
            n_timestreams = np.size(data_0, 0)
            data_array = np.zeros((len(data_files)*n_timestreams, self.Np))
            for i, file in enumerate(data_files):
                data_file = np.load(file)
                for j in range(n_timestreams):
                    data_array[i*n_timestreams+j,:] += data_file[j]
        else:
        #Temporary for testing                            #what is this for?
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
        
        #Normalize RMS to RMS of first timestream                          Why?
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
            
    
class RFIDetect:
    def __init__(self, Np, z_dim = 16, hidden_dim = 1024, nworkers = 0, Nepochs = 30):
        self.Np = Np
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.nworkers = nworkers           #what are these? nworkers?
        self.Nepochs = Nepochs
        
        self.wrapper = 'example.ipynb'
        self.code = 'rfi_ml.py'
        self.save_folder = 'rfi_ml/'
        os.makedirs(self.save_folder, exist_ok=True)
        self.save_time = str(datetime.datetime.now()).split('.')[0].replace(' ','_').replace(':','-')
        
        
        os.system('scp ./' + self.wrapper + ' ' + self.save_folder + '/' + self.save_time + '_' + self.wrapper)
        os.system('scp ../' + self.code + ' ' + self.save_folder + '/' + self.save_time + '_' + self.code)

        # Encoder network
    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, z_dim):
            super(RFIDetect.Encoder, self).__init__()

            self.main = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
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
                nn.Linear(hidden_dim, out_dim, bias=False),
            )

        def forward(self, x):
            out = self.main(x)
            return out



    def train(self, train_array, gauss_fact=torch.ones(1), batch_size = 32, lr=0.0002, betas=(0.5, 0.999)):
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
                             
                s = s[0].float().cuda()
                
                # encode-decode
                recons_out = self.netD(self.netE(s))
                
                # loss
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
        
    def evaluate(self, test_array, gauss_fact=torch.ones(1), lamb=0):
            
        self.netE.eval()
        self.netD.eval()
        
        recons_out = []       
        with torch.no_grad():
            sigs = torch.from_numpy(test_array).float().cuda()
            recons_out = self.netD(self.netE(sigs))
          
        self.rms = np.std(sigs.cpu().numpy(), axis=1)
        self.avg_rms = np.mean(self.rms)

        print("Epochs: ",self.Nepochs)
        print("N tests: ", test_array.shape[0])
        print("Length of timestream: ", self.Np)
        print("Input Timestream RMS: ", self.rms)
        print("Avg RMS for all tests: ", self.avg_rms, "\n") 
        
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
            Plots that compare raw signal to cleaned signal
            """
            
            fig1 = plt.figure(figsize=(20,10))
            
            #Overplot
            ax = fig1.add_subplot(1,2,1)
            plt.plot(test_array[test_int,:], color='steelblue')
            plt.plot(recons_out[test_int,:].cpu().numpy(), color='orangered')
            plt.xlabel('Something', fontsize=11)
            plt.ylabel('Something else', fontsize=11)
            ax.legend(['Timestream In','RFI Recovered'])

            #RFI cleaned
            ax = fig1.add_subplot(1,2,2)
            plt.plot(test_array[test_int].cpu().numpy(), color='steelblue')
            plt.plot(sig[test_int,:]-recons_out[test_int,:].cpu().numpy(), color='forestgreen')
            plt.ylabel('Amplitude', fontsize=11)
            plt.xlabel('Sample Length', fontsize=11)
            ax.legend(['Input Signal','RFI Subtracted Timestream'])

            save_filename = self.save_time + 'contrast' + '_test_' + str(test_int) + '.png'
            save_path = os.path.join(self.save_folder, save_filename)
            print('Saving file...{}'.format(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            
            fig1.clf #clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
            plt.close(fig1)


            """
            Power spectra plots

            I wonder what sort of plots we can make that are relevant for a CMB collaboration. Maybe I can find something useful online (I'll consult github and videos)
            """


            
            
            """
            Tensor Diagram will be saved separately
            """
            


            
