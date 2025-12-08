import numpy as np
#from astropy.io import fits
import torch
from torch.utils.data import Dataset

from .simulation import get_sim

import sys,time,os
# sys.path.append(os.path.abspath("../configs"))
# from configs import config
# import config


class BaseDataset(Dataset):
    """
    Base class for all datasets providing common functionality.
    """
    def __init__(self, config=None):
        self.config = config

    def compute_noise(self, noisy_im, clean_im):
        """
        Compute the noise level and mean background from a noisy image and a clean image.

        Args:
            noisy_im (np.ndarray): The image with noise.
            clean_im (np.ndarray): The clean image (without noise).

        Returns:
            tuple: A tuple containing:
                - sig_sky (float): Standard deviation of the noise.
                - mean_sky (float): Mean of the noise.
        """
        noise = noisy_im - clean_im
        sig_sky = np.std(noise)
        mean_sky = np.mean(noise)
        return sig_sky, mean_sky


#### Dataset for CNN training. You need to customize it
class ShapeDataset(BaseDataset):
    """
    Dataset for training a CNN to measure galaxy shapes.
    """
    
    def __init__(self, gal_pars, psf_pars, config):
        """
        Initialize the ShapeDataset.

        Args:
            gal_pars (dict): Dictionary of galaxy parameters.
            psf_pars (dict): Dictionary of PSF parameters.
            config (Config): Configuration object.
        """
        super().__init__(config)
        self.gal_pars = gal_pars
        self.psf_pars = psf_pars

    def __len__(self):
        return self.gal_pars["e1"].shape[0]
    
    def __set_pars(self, idx):
        
        self.param_gal = {k: self.gal_pars[k][idx] for k in self.gal_pars}
        self.param_psf = {k: self.psf_pars[k][idx] for k in self.psf_pars}
        

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - 'gal_image': Galaxy image with noise.
                - 'psf_image': PSF image.
                - 'label': Normalized labels (e1, e2, hlr, mag).
                - 'snr': Signal-to-noise ratio.
                - 'id': Sample index.
        """
        self.__set_pars(idx)
        
        gal_image, clean_gal, psf_image, label_ = get_sim(
                gal_param=self.param_gal,
                psf_param=self.param_psf,
                config=self.config,
                shear=None,
            )
        
        label = np.array([label_[0]/np.max(self.gal_pars["e1"]),
                          label_[1]/np.max(self.gal_pars["e2"]),
                          label_[2]/np.max(self.gal_pars["hlr_disk"]),
                          label_[3]/np.max(self.gal_pars["mag_i"])])
        
        sig_sky,_ = self.compute_noise(gal_image,clean_gal)
        snr = np.sqrt(np.sum(np.power(clean_gal,2))/sig_sky**2)

        return {'gal_image': gal_image[None], 
                'psf_image': psf_image[None], 
                'label': label,
                'snr': snr,
                'id': idx}
    

### Dataset for shear measurement
class ShearDataset(BaseDataset):
    """
    Dataset for shear measurement tasks.
    """
    def __init__(self, shear_set, gal_set, psf_set, config):
        """
        Initialize the ShearDataset.

        Args:
            shear_set (dict): Dictionary containing shear values.
            gal_set (dict): Dictionary of galaxy parameters.
            psf_set (dict): Dictionary of PSF parameters.
            config (Config): Configuration object.
        """
        super().__init__(config)
        self.shear_set = shear_set
        self.gal_set = gal_set
        self.psf_set = psf_set

    def __len__(self):
        return self.gal_set["hlr_disk"].shape[0]*self.gal_set["hlr_disk"].shape[1]
    
    def __set_pars(self, idx):
        
        case_idx = idx//self.gal_set["hlr_disk"].shape[1]
        real_idx = idx%self.gal_set["hlr_disk"].shape[1]
        
        self.param_gal = {k: self.gal_set[k][case_idx,real_idx] for k in self.gal_set}
        self.shear = self.shear_set['shear'][case_idx,:]
        self.param_psf = {k: self.psf_set[k][case_idx,real_idx] for k in self.psf_set}

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - 'gal_image': Galaxy image with noise.
                - 'psf_image': PSF image.
                - 'label': Normalized labels.
                - 'snr': Signal-to-noise ratio.
                - 'id': Sample index.
        """
        self.__set_pars(idx)

        gal_image, clean_gal, psf_image, label_ = get_sim(
            self.param_gal,
            self.param_psf,
            config=self.config,
            shear=self.shear,
        )
        
        # note the normalization here
        label = np.array([label_[0],
                  label_[1],
                  label_[2]/1.2,
                  label_[3]/25])
        
        sig_sky,_ = self.compute_noise(gal_image,clean_gal)
        snr = np.sqrt(np.sum(np.power(clean_gal,2))/sig_sky**2)
        
        return {'gal_image': gal_image[None], 
                'psf_image': psf_image[None], 
                'label': label, 
                'snr': snr,
                'id': idx}
    

### Dataset for NN calibration
class CaliDataset(BaseDataset):
    """
    Dataset for Neural Network calibration.
    """
    def __init__(self, shear_set, gal_set, psf_set, config):
        """
        Initialize the CaliDataset.

        Args:
            shear_set (dict): Dictionary containing shear values.
            gal_set (dict): Dictionary of galaxy parameters.
            psf_set (dict): Dictionary of PSF parameters.
            config (Config): Configuration object.
        """
        super().__init__(config)
        self.shear_set = shear_set
        self.gal_set = gal_set
        self.psf_set = psf_set

    def __len__(self):
        return self.gal_set["e1"].shape[0]*self.gal_set["e2"].shape[1]
    
    def __set_pars(self, idx):
        
        case_idx = idx//self.gal_set["e1"].shape[1]
        real_idx = idx%self.gal_set["e1"].shape[1]
        
        self.param_gal = {k: self.gal_set[k][case_idx,real_idx] for k in self.gal_set}
        self.shear = self.shear_set['shear'][case_idx,:]
        self.param_psf = {k: self.psf_set[k][case_idx,real_idx] for k in self.psf_set}

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - 'gal_image': Galaxy image with noise.
                - 'psf_image': PSF image.
                - 'label': Normalized labels.
                - 'snr': Signal-to-noise ratio.
                - 'id': Sample index.
        """
        self.__set_pars(idx)

        gal_image, clean_gal, psf_image, label_ = get_sim(
            self.param_gal,
            self.param_psf,
            config=self.config,
            shear=self.shear,
        )
        
        # note the normalization here
        label = np.array([label_[0],
                  label_[1],
                  label_[2],
                  label_[3]],
                  dtype=np.float32)
        
        sig_sky,_ = self.compute_noise(gal_image,clean_gal)
        snr = np.sqrt(np.sum(np.power(clean_gal,2))/sig_sky**2)

        info = np.array([snr,
                        self.shear[0],
                        self.shear[1]],
                        dtype=np.float32)
        
        return {'gal_image': gal_image[None], 
                'psf_image': psf_image[None], 
                'label': label, 
                'info': info,
                'id': idx}
    

### Dataset for NN calibration training
class NNDataset(BaseDataset):
    """
    Dataset for training the calibration Neural Network.
    """
    def __init__(self, dataset, config=None):
        """
        Initialize the NNDataset.

        Args:
            dataset (dict): Dictionary containing 'prediction' and 'true_shear'.
            config (Config): Configuration object.
        """
        super().__init__(config)
        self.classes_frame = dataset
        self.case_size = self.classes_frame['prediction'].shape[0]
        self.real_size = self.classes_frame['prediction'].shape[1]

    def __len__(self):
        return self.case_size*self.real_size

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - 'input': Measured galaxy parameters (prediction).
                - 'label': True shear values.
                - 'id': Sample index.
        """
        #print(idx)
        
        idx_case = idx//self.real_size
        idx_real = idx%self.real_size
        
        measured_gal = self.classes_frame['prediction'][idx_case,idx_real,:].astype(np.float32)
        shear = self.classes_frame['true_shear'][idx_case].astype(np.float32)

        return {'input': measured_gal, 
                'label': shear, 
                'id': idx}


class RegDataset(BaseDataset):
    """
    A simple regression dataset for flat input and output arrays.
    """
    
    def __init__(self, inputs, labels, config=None):
        """
        Initialize the RegDataset.

        Args:
            inputs (np.ndarray or torch.Tensor): Input data of shape (N, in_features).
            labels (np.ndarray or torch.Tensor): Target labels of shape (N, out_features).
            config (Config, optional): Configuration object.
        """
        super().__init__(config)
        self.inputs = inputs
        self.labels = labels
        self.real_size = 1 # Treat as independent samples (no multi-realization structure)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing 'input', 'label', and 'id'.
        """
        return {
            'input': torch.tensor(self.inputs[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
            'id': idx
        }