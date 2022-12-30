import numpy as np
#from astropy.io import fits
from torch.utils.data import Dataset

from .simulation import get_sim

import sys,time,os
# sys.path.append(os.path.abspath("../configs"))
# from configs import config
import config


def compute_noise(noisy_im,clean_im):

    noise = noisy_im - clean_im

    # noise = noisy_im[0:2,:]
    # noise = np.vstack((noise, noisy_im[-2:,:]))

    sig_sky = np.std(noise)
    mean_sky = np.mean(noise)
    #snr = np.sqrt(np.sum(np.power(clean_im,2))/sig_sky**2)
    return sig_sky, mean_sky


#### Dataset for training. You need to customize it
class ShapeDataset(Dataset):
    
    def __init__(self, gal_pars, psf_pars):
        
        self.gal_pars = gal_pars
        self.psf_pars = psf_pars

    def __len__(self):
        return self.gal_pars["e1"].shape[0]
    
    def __set_pars(self, idx):
        
        self.param_gal = {}
        self.param_gal["e1"] = self.gal_pars["e1"][idx]
        self.param_gal["e2"] = self.gal_pars["e2"][idx]
        self.param_gal["hlr_disk"] = self.gal_pars["hlr_disk"][idx]
        self.param_gal["mag_i"] = self.gal_pars["mag_i"][idx]

        self.param_psf = {}
        # param_psfs['size'] = self.classes_frame[idx,7]
        # param_psfs['e1'] = self.classes_frame[idx,4]
        # param_psfs['e2'] = self.classes_frame[idx,5]
        self.param_psf['randint'] = self.psf_pars["randint"][idx]
        

    def __getitem__(self, idx):
        
        self.__set_pars(idx)
        
        gal_image, clean_gal, psf_image, label_ = get_sim(
                gal_param=self.param_gal,
                psf_param=self.param_psf,
                shear=None,
            )
        
        label = np.array([label_[0]/np.max(self.gal_pars["e1"]),
                          label_[1]/np.max(self.gal_pars["e2"]),
                          label_[2]/np.max(self.gal_pars["hlr_disk"]),
                          label_[3]/np.max(self.gal_pars["mag_i"])])
        
        sig_sky,_ = compute_noise(gal_image,clean_gal)
        snr = np.sqrt(np.sum(np.power(clean_gal,2))/sig_sky**2)

        return {'gal_image': gal_image[None], 
                'psf_image': psf_image[None], 
                'label': label,
                'snr': snr,
                'id': idx}
    
    
class ShearDataset(Dataset):
    def __init__(self, shear_set, gal_set):
        
        self.shear_set = shear_set
        self.gal_set = gal_set

    def __len__(self):
        return self.gal_set["hlr_disk"].shape[0]*self.gal_set["hlr_disk"].shape[1]
    
    def __set_pars(self, idx):
        
        case_idx = idx//self.gal_set["hlr_disk"].shape[1]
        real_idx = idx%self.gal_set["hlr_disk"].shape[1]
        
        self.param_gal = {}
        self.param_gal["hlr_disk"] = self.gal_set["hlr_disk"][case_idx,real_idx]
        self.param_gal["mag_i"] = self.gal_set["mag_i"][case_idx,real_idx]
        self.param_gal["e1"] = self.gal_set["e1"][case_idx,real_idx]
        self.param_gal["e2"] = self.gal_set["e2"][case_idx,real_idx]
        
        self.shear = self.shear_set['shear'][case_idx,:]

        self.param_psf = {}
        self.param_psf['randint'] = self.gal_set['randint'][case_idx,real_idx]

    def __getitem__(self, idx):
        
        self.__set_pars(idx)

        gal_image, clean_gal, psf_image, label_ = get_sim(
            self.param_gal,
            self.param_psf,
            shear=self.shear,
        )

        label = np.array([label_[0]/0.6,
                  label_[1]/0.6,
                  label_[2]/1.2,
                  label_[3]/25])
        
        sig_sky,_ = compute_noise(gal_image,clean_gal)
        snr = np.sqrt(np.sum(np.power(clean_gal,2))/sig_sky**2)
        
        return {'gal_image': gal_image[None], 
                'psf_image': psf_image[None], 
                'label': label, 
                'snr': snr,
                'id': idx}
    
