import numpy as np
import os, datetime, pickle, glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

if not torch.cuda.is_available():
    print ("Warning: I see no CUDA, this will be slow!")
 

#Get output of NN at intermediate steps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook    
    
class BMXLoader:
    def __init__(self, Np=2048, freq='0000'):
        self.Np = Np
        self.freq = freq
        
    def loadData(self, data_dir):
        data_files = glob.glob(data_dir+'**/*_'+self.freq+'.npy')
        print(data_files)
        
        data_0 = np.load(data_files[0])
        n_timestreams = np.size(data_0, 0)
        data_array = np.zeros((len(data_files)*n_timestreams, self.Np), dtype=float32)
        for i in range(data_files):
            data_file = np.load(data_files[i])
            for j in range(n_timestreams):
                data_array[i*n_timestreams+j,:] += data_file[j]
            
        return data_array      

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
        rfi_pwr = np.sum((rfi[round(pos-3*sigma):round(pos+3*sigma)])**2)/(6*sigma)
        return rfi, rfi_pwr

    def getNonGaussianNonlocalized(self, freq=(0.2, 0.5), ampl=(0.05, 0.05), Pflip=1.0):
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
    
class RFIDetect:
    
    # Decoder nework
    class Decoder(nn.Module):                                               #consider moving
        def __init__(self, z_dim, hidden_dim, hidden_dim_2, out_dim):
            super(RFIDetect.Decoder, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(z_dim, hidden_dim_2),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Linear(hidden_dim_2, hidden_dim),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Linear(hidden_dim, out_dim, bias=False),
            )


        def forward(self, x):
            
            out = self.main(x)
            return out
        
    # Encoder network
    class Encoder(nn.Module):                                               #consider moving
        def __init__(self, input_dim, hidden_dim, hidden_dim_2, z_dim):
            super(RFIDetect.Encoder, self).__init__()

            self.main = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Linear(hidden_dim, hidden_dim_2),
                nn.LeakyReLU(0.02, inplace=False),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim_2, z_dim),
            )
  
            
        def forward(self, x):
            
            out = self.main(x)
            return out

    def __init__(self, Np, z_dim = 16, hidden_dim = 1024, hidden_dim_2 = 512, nworkers = 0, Nepochs = 30): #consider moving up to top
        self.Np = Np
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_2 = hidden_dim_2
        self.nworkers = nworkers
        self.Nepochs = Nepochs
        
        self.wrapper = 'example.ipynb'
        self.code = 'rfi_mlM.py'
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

    def train(self, g_train_array, ng_train_array, gauss_fact=torch.ones(1), lamb=0, batch_size = 32, lr=0.0002, betas=(0.5, 0.999)):
        
        s_trainloader = DataLoader(                                            
            torch.utils.data.TensorDataset(g_train_array + ng_train_array),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.nworkers,
            pin_memory=True,
            drop_last=True,
        )
        
        if not hasattr(self,"netD"):
            self.netD = self.Decoder(
                z_dim=self.z_dim, hidden_dim=self.hidden_dim, hidden_dim_2=self.hidden_dim_2, out_dim=self.Np
            ).cuda()
            self.netE = self.Encoder(
                input_dim=self.Np, hidden_dim=self.hidden_dim, hidden_dim_2=self.hidden_dim_2, z_dim=self.z_dim
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
            for i, s in enumerate(s_trainloader): 
                # set to train mode
                self.netE.train()
                self.netD.train()
                             
                s = s[0].float().cuda() #moves to GPU
                
                #gaussianized = torch.stack([gauss_fact * self.Gaussianize(sig.cpu()) for sig in s]).cuda().float() #Normalization option component
                #modsig = np.sqrt(1-lamb**2)*s + lamb*gaussianized #Normalization option
                
                # encode-decode
                recons_out = self.netD(self.netE(s))
                #recons_out = self.netD(self.netE(modsig)) #Normalization option
                
                # loss
                loss = recons_criterion(s, recons_out)
                #loss = recons_criterion(modsig - recons_out, lamb*gaussianized) #Normalization option
                
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
                 
    def evaluate(self, g_test_array, ng_test_array, rfi_pwr, gauss_fact=torch.ones(1), lamb=0):
            
        self.netE.eval()
        self.netD.eval()

        self.netE.main[0].register_forward_hook(get_activation('layerE0'))
        self.netE.main[2].register_forward_hook(get_activation('layerE2'))
        self.netE.main[4].register_forward_hook(get_activation('layerE4'))
        #self.netE.main[7].register_forward_hook(get_activation('layerE7'))        
        self.netD.main[0].register_forward_hook(get_activation('layerD0'))
        self.netD.main[2].register_forward_hook(get_activation('layerD2'))
        self.netD.main[4].register_forward_hook(get_activation('layerD4'))
        #self.netD.main[6].register_forward_hook(get_activation('layerD6'))
        
        recons_out = []       
        with torch.no_grad(): #turning off gradient computation for NN evaluation
            sigs = g_test_array.float().cuda() + ng_test_array.float().cuda()
            
            #gaussianized = torch.stack([gauss_fact * torch.from_numpy(self.Gaussianize(sig.cpu().numpy())) for sig in sigs]).cuda().float() #Normalization option  component
            #modsig = np.sqrt(1-lamb**2)*sigs + lamb*gaussianized #Normalization option                 
            
            #recons_out = self.netD(self.netE(modsig)) #Normalization option
            recons_out = self.netD(self.netE(sigs))
            
                 
        self.rms = np.sqrt(np.sum((recons_out.cpu().numpy()-ng_test_array.cpu().numpy())**2, axis=1)/self.Np)
        self.avg_rms = np.sum(self.rms)/ng_test_array.cpu().numpy().shape[0]
        self.rfi_pwr_remaining = np.sum((ng_test_array.cpu().numpy()-recons_out.cpu().numpy())**2, axis=1)
        self.frac_rfi_pwr = self.rfi_pwr_remaining/np.sum(ng_test_array.cpu().numpy()**2, axis=1)
        self.frac_sig_pwr = self.rfi_pwr_remaining/np.sum(g_test_array.cpu().numpy()**2, axis=1)
        self.avg_rfi_pwr_remain = np.sum(self.frac_rfi_pwr)/ng_test_array.cpu().numpy().shape[0]
        self.avg_gauss = np.sum(np.abs(g_test_array.cpu().numpy()),axis=None)/(self.Np*g_test_array.cpu().numpy().shape[0])
        self.rfi_rms = np.sqrt(np.sum((ng_test_array.cpu().numpy())**2, axis=1)/self.Np)
        self.rfi_avg_rms = np.sum(self.rfi_rms)/ng_test_array.cpu().numpy().shape[0]
        self.rfi_avg_pwr = np.sum(rfi_pwr)/rfi_pwr.shape[0]
        self.g_rms = np.sqrt(np.sum((g_test_array.cpu().numpy())**2, axis=1)/self.Np)
        self.g_avg_rms = np.sum(self.g_rms)/g_test_array.cpu().numpy().shape[0]
        self.sn = (self.rfi_avg_rms/self.g_avg_rms)**2
        self.sn_2 = self.rfi_avg_pwr/(self.g_avg_rms)**2

        
        sig = (g_test_array.cpu() + ng_test_array.cpu()).numpy()
        self.Nfft = ng_test_array.cpu().numpy().shape[1] // 2 + 1
        self.sky_pwr_spec = (1/self.Nfft**2)*np.abs(np.fft.rfft(g_test_array))**2
        self.recon_pwr_spec = (1/self.Nfft**2)*np.abs(np.fft.rfft(sig-recons_out.cpu().numpy()))**2
        self.sig_pwr_spec = (1/self.Nfft**2)*np.abs(np.fft.rfft(sig))**2

        #FIND A WAY TO SAVE ALL THE VALUES BELOW B/C IT DOESN'T SAVE

        print("Epochs: ",self.Nepochs)
        print("N tests: ", ng_test_array.cpu().numpy().shape[0])
        print("Length of timestream: ", self.Np)
        print("Timestream Obs-Expect RMS: ", self.rms)
        print("Avg RMS for all tests: ", self.avg_rms, "\n")    
        print("Fraction of RFI power remaining: ", self.frac_rfi_pwr)
        print("Remainder as fraction of signal power: ", self.frac_sig_pwr)
        print("Avg of RFI power remaining: ",self.avg_rfi_pwr_remain)
        print("Input RFI Timestream RMS: ", self.rfi_rms)
        print("Avg RFI RMS: ", self.rfi_avg_rms)  
        print("Input Gaussian Timestream RMS: ", self.g_rms)
        print("Avg Gaussian RMS: ", self.g_avg_rms)  
        print("Signal to Noise Ratio: ", self.sn)  
        print("Signal to Noise Ratio 2: ", self.sn_2)     
        #print("Avg amp of Gaussian signal: ",self.avg_gauss)
        
        save_filename = self.save_time + '_epoch_' + str(self.Nepochs).zfill(7) + '_eval_data'
        save_path = os.path.join(self.save_folder, save_filename)
        #with open(save_path, 'wb') as file:
        #    pickle.dump(self, file)

        return recons_out, activation

    def indexing(self, recons_out, threshold=0.7):
    
    
        recons_contaminated = []
        recons_clean = []
        
        for index, samples in enumerate(recons_out):

            # Power of reconstructed output

            recons_power = np.sum((samples.cpu().numpy())**2)                # NEW NEW NEW
            
            if recons_power > threshold:
                recons_contaminated.append(index)
            else:
                recons_clean.append(index)


            # Define file names
        save_contaminated = self.save_time + '_contaminated' + '_index' + '.txt'
        save_clean = self.save_time + '_clean' + '_index' + '.txt'
    
        # Define file paths
        save_path = os.path.join(self.save_folder, save_contaminated)
        save_path1 = os.path.join(self.save_folder, save_clean)
        
        # Print file paths
        print(f'Saving file... {save_path}')
        print(f'Saving file... {save_path1}')

        # Print the number of clean samples
        print('number of clean samples:', len(recons_clean))
        print('number of contaminated samples:', len(recons_contaminated))
        print('false positive rate:', len(recons_clean) / len(recons_out))
        
        # Check if the folder exists, create if it doesn't
        if not os.path.exists(self.save_folder):
            print(f'Folder {self.save_folder} does not exist. Creating it.')
            os.makedirs(self.save_folder)
        
        # Save the contaminated indices to file
        with open(save_path, 'w') as file:
            for index in recons_contaminated:
                file.write(f"{index}\n")
                print(f'Writing contaminated index: {index}')  # Debug: Log each index written
        
        # Save the clean indices to file
        with open(save_path1, 'w') as file:
            for index in recons_clean:
                file.write(f"{index}\n")
                print(f'Writing clean index: {index}')  # Debug: Log each index written
            
        
    
    def plot_eval(self, recons_out, g_test_array, ng_test_array, activation):
            #Test diagnostics at end of epoch     
            self.Nbins = 32
            time = range(self.Np)
            sig = (g_test_array.cpu() + ng_test_array.cpu()).numpy()
            chi_sqr = np.zeros(len(g_test_array))
            med_frac_diff = np.zeros(len(g_test_array))
            self.sky_spec_binned = np.zeros([self.Nbins, len(g_test_array)])
            self.recon_spec_binned = np.zeros([self.Nbins, len(g_test_array)])
            self.sig_spec_binned = np.zeros([self.Nbins, len(g_test_array)])
            self.pwr_spec_err = np.zeros([self.Nbins, len(g_test_array)])
            
            for test_int in range(len(g_test_array)):
                gauss = self.Gaussianize(sig[test_int,:])
                
                """ Standard Plots. Disabled for Ntest = 1000 run.
                """
     
                #Plot components separately
                plt.figure(figsize=(17,10)) 
                ax = plt.subplot(2,2,1)
                plt.plot(time, ng_test_array[test_int].cpu().numpy()) # ng
                ax.set_title("RFI Signal ($T_{ng}$)")
    
                ax = plt.subplot(2,2,2)
                plt.plot(time, g_test_array[test_int].cpu().numpy()) # g
                ax.set_title("Gaussian Distributed Sky Signal ($T_g$)")
    
                ax = plt.subplot(2,2,3)
                plt.plot(time, sig[test_int,:]) # g + ng 
                ax.set_title("Combined Signal (S)")
    
                ax = plt.subplot(2,2,4)
                plt.plot(time, recons_out[test_int,:].cpu().numpy()) # output
                plt.plot(ng_test_array[test_int].cpu().numpy()-recons_out[test_int,:].cpu().numpy()) #resid rfi
                ax.set_title("Network Output (~$T_{ng}$)")
                ax.legend(['Recovered RFI','Residual RFI'])
    
                save_filename = self.save_time + '_epoch_' + str(self.Nepochs).zfill(7) + '_test_' + str(test_int) + '.pdf'
                save_path = os.path.join(self.save_folder, save_filename)
                print('Saving file...{}'.format(save_path))
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                
                #Overplot
                plt.plot(sig[test_int,:])
                plt.plot(ng_test_array[test_int].cpu().numpy())
                plt.plot(recons_out[test_int,:].cpu().numpy())
                plt.legend(['Test In','RFI','Recovered'])
        
                save_filename = self.save_time + '_overplot_test_' + str(test_int) + '.pdf'
                save_path = os.path.join(self.save_folder, save_filename)
                print('Saving file...{}'.format(save_path))
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                
                #Residual RFI
                plt.plot(ng_test_array[test_int].cpu().numpy())
                plt.plot(ng_test_array[test_int].cpu().numpy()-recons_out[test_int,:].cpu().numpy())
                plt.legend(['Input RFI','Residual RFI'])
        
                save_filename = self.save_time + '_residual_rfi_test_' + str(test_int) + '.pdf'
                save_path = os.path.join(self.save_folder, save_filename)
                print('Saving file...{}'.format(save_path))
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()            
                
                #RFI cleaned
                plt.plot(g_test_array[test_int].cpu().numpy())
                plt.plot(sig[test_int,:]-recons_out[test_int,:].cpu().numpy())
                plt.legend(['True Sky Signal','RFI Subtracted Timestream'])
                plt.title('RFI Cleaned Timestream')
                plt.ylabel('Amplitude')
                plt.xlabel('Sample')
    
                save_filename = self.save_time + '_RFI_subtracted_test_' + str(test_int) + '.pdf'
                save_path = os.path.join(self.save_folder, save_filename)
                print('Saving file...{}'.format(save_path))
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                
                #difference
                diff = g_test_array[test_int].cpu().numpy() - (sig[test_int,:]-recons_out[test_int,:].cpu().numpy())
                plt.plot(g_test_array[test_int].cpu().numpy())
                plt.plot(diff)
                plt.legend(['True - Recovered Timestream'])
        
                save_filename = self.save_time + '_diff_test_' + str(test_int) + '.png'
                save_path = os.path.join(self.save_folder, save_filename)
                print('Saving file...{}'.format(save_path))
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()  
                
                #frac
                frac = diff/g_test_array[test_int].cpu().numpy()
                plt.semilogy(abs(frac))
                plt.legend(['Fractional Error Timestream'])
        
                save_filename = self.save_time + '_frac_err_test_' + str(test_int) + '.png'
                save_path = os.path.join(self.save_folder, save_filename)
                print('Saving file...{}'.format(save_path))
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()      
                
                from statistics import median
                med_frac_diff[test_int] = median(frac)
                print('Median Frac Diff: ', med_frac_diff[test_int])
                
                
                #Power spectra comparison
                self.freqs_per_bin = int((self.Nfft-1)/self.Nbins)
                self.sky_spec_binned[:,test_int] = self.sky_pwr_spec[test_int,1:].reshape((self.Nbins,self.freqs_per_bin)).mean(axis=1)
                self.recon_spec_binned[:,test_int] = self.recon_pwr_spec[test_int,1:].reshape((self.Nbins,self.freqs_per_bin)).mean(axis=1)
                self.sig_spec_binned[:,test_int] = self.sig_pwr_spec[test_int,1:].reshape((self.Nbins,self.freqs_per_bin)).mean(axis=1)
                self.pwr_spec_err[:,test_int] = self.recon_spec_binned[:,test_int]*np.sqrt(1/self.freqs_per_bin)*np.sqrt(1/2)
                
                chi_sqr[test_int] = ((self.sky_spec_binned[:,test_int] - self.recon_spec_binned[:,test_int])**2 / self.pwr_spec_err[:,test_int]**2).mean()
                #print('Chi Squared: ', chi_sqr[test_int])
                #print('DoF: ', self.Nbins-1)
                from scipy.stats import chi2
                #print('P Value: ', 1-chi2.cdf(chi_sqr[test_int],self.Nbins-1))
                #print('1-P Value: ', chi2.cdf(chi_sqr[test_int],self.Nbins-1))
        
                """
                plt.plot(range(self.Nbins), self.sky_spec_binned[:,test_int])
                plt.errorbar(range(self.Nbins), self.recon_spec_binned[:,test_int], yerr=self.pwr_spec_err[:,test_int], fmt='o')
                #plt.plot(range(self.Nbins), self.sky_pwr_spec[test_int,:])
                #plt.errorbar(range(self.Nfft), self.recon_pwr_spec[test_int,:], yerr=pwr_spec_err, fmt='o')
                plt.legend(['True Sky Power Spectrum','Recovered Sky Power Spectrum'])
        
                save_filename = self.save_time + '_power_spectra_test_' + str(test_int) + '.pdf'
                save_path = os.path.join(self.save_folder, save_filename)
                print('Saving file...{}'.format(save_path))
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()    
                """
            
            #Plot average sky, rfi contaminated, and recovered spectra 
            sky_spec_avg = self.sky_spec_binned.mean(axis=1)
            recon_spec_avg = self.recon_spec_binned.mean(axis=1)
            sig_spec_avg = self.sig_spec_binned.mean(axis=1)
            pwr_spec_err_avg = recon_spec_avg*np.sqrt(1/self.freqs_per_bin)*np.sqrt(1/len(g_test_array))*np.sqrt(1/2)
            plt.plot(range(self.Nbins), sig_spec_avg, color='orange')
            plt.plot(range(self.Nbins), sky_spec_avg, color='blue')
            plt.errorbar(range(self.Nbins), recon_spec_avg, yerr=pwr_spec_err_avg, fmt='o', markersize=4, color='green')
            plt.yscale('log')        
            #plt.plot(range(self.Nbins), self.sky_pwr_spec[test_int,:])
            #plt.errorbar(range(self.Nfft), self.recon_pwr_spec[test_int,:], yerr=pwr_spec_err, fmt='o')
            plt.legend(['RFI Contaminated Power Spectrum','True Sky Power Spectrum','Recovered Sky Power Spectrum'])
            plt.title('Average of ' + str(len(g_test_array)) + ' Recovered Power Spectra')
            plt.ylabel('Power')
            plt.xlabel('Frequency Bin')
            
            save_filename = self.save_time + '_power_spectra_test_avg_n_' + str(len(g_test_array)) + '.pdf'
            save_path = os.path.join(self.save_folder, save_filename)
            print('Saving file...{}'.format(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()  
     
            #Plot with no RFI trace
            plt.plot(range(int(self.Nbins)), sky_spec_avg[:int(self.Nbins)])
            plt.errorbar(range(int(self.Nbins)), recon_spec_avg[:int(self.Nbins)], yerr=pwr_spec_err_avg[:int(self.Nbins)], fmt='o')
            plt.yscale('log')
            plt.legend(['True Sky Power Spectrum','Recovered Sky Power Spectrum'])
    
            save_filename = self.save_time + '_power_spectra_test_avg_n_' + str(len(g_test_array)) + '_no_sig_trace.pdf'
            save_path = os.path.join(self.save_folder, save_filename)
            print('Saving file...{}'.format(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()  
            
            #Plot residuals
            #plt.plot(range(int(self.Nbins)), sky_spec_avg*0)
            plt.errorbar(range(int(self.Nbins)), (recon_spec_avg-sky_spec_avg)/sky_spec_avg, yerr=pwr_spec_err_avg/sky_spec_avg, fmt='o')
            #plt.yscale('log')
            #plt.legend(['True Sky Power Spectrum','Recovered Power Spectrum Residuals'])
            plt.title('Fractional Recovered Power Spectrum Residuals')
            #plt.ylabel('$(P_\mathrm{recov}-P_\mathrm{sky}) \ / \ P_\mathrm{sky}$') #checked
            plt.xlabel('Frequency Bin')
    
            save_filename = self.save_time + '_power_spectra_residuals_n_' + str(len(g_test_array)) + '.pdf'
            save_path = os.path.join(self.save_folder, save_filename)
            print('Saving file...{}'.format(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()  
            
            print('Mean Chi Squared: ', chi_sqr.mean())
            print('Mean Chi Squared per Degree Freedom: ', chi_sqr.mean()/((self.Nbins-1))) 
            print('P Value: ', 1-chi2.cdf(chi_sqr[test_int],(self.Nbins-1)))
            print('1-P Value: ', chi2.cdf(chi_sqr[test_int],(self.Nbins-1)))
            print('Mean Fractional Difference: ', med_frac_diff.mean())
        
            """
            #S/N plot
            sn = [0.25, 0.3, 0.4, 0.5, 1.0, 2.0]
            false_neg_0 = [0.9, 0.79, 0.43, 0.08, 0.0, 0.0]
            #false_neg_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            plt.plot(sn, false_neg_0)
            #plt.plot(sn, false_neg_1)
            plt.xlabel("Signal-to-Noise Ratio")
            plt.ylabel("False Negative Rate")
            #plt.legend(['Localized RFI','Long Period RFI'])
        
            save_filename = self.save_time + '_false_negative_rates.pdf'
            save_path = os.path.join(self.save_folder, save_filename)
            print('Saving file...{}'.format(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            """
                
            #rand int seed check
            #print('Numpy rand int check: ',np.random.uniform(0,1,1))
            #print('Torch rand int check: ',torch.rand(1))