# import config
from forklens import train
from forklens.dataset import ShapeDataset
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# Load data catalog
with fits.open('../data/disk_gal_catalog_cut_25_1.2.fits') as f:
    mag_cat = f[0].data
    size_cat = f[3].data
    
    
# Generate data
def dataframe(num, seed=12345):
    
    num = int(num/2)
    rng1 = np.random.RandomState(seed)
    Gal_Hlr   = size_cat[rng1.randint(0,mag_cat.shape[0],size=num)]
    Gal_Hlr   = np.concatenate((Gal_Hlr, Gal_Hlr), axis=0)
    Gal_Mag   = mag_cat[rng1.randint(0,mag_cat.shape[0],size=num)]
    Gal_Mag   = np.concatenate((Gal_Mag, Gal_Mag), axis=0)
    Gal_e     = rng1.random((num,2))*(-0.6-0.6)+0.6
    Gal_e     = np.concatenate((Gal_e, Gal_e*-1), axis=0)

    rng2 = np.random.RandomState(seed+1)
    PSF_randint = rng2.randint(0,10000,size=num)
    PSF_randint = np.concatenate((PSF_randint, PSF_randint), axis=0)

    # Dataframe        = np.zeros((num*2, 5))
    # Dataframe[:,0:2] = Gal_e
    # Dataframe[:,2]   = Gal_Hlr
    # Dataframe[:,3]   = Gal_Mag
    # Dataframe[:,4] = PSF_randint
    
    gal_pars = {}
    gal_pars["e1"] = Gal_e[:,0]
    gal_pars["e2"] = Gal_e[:,1]
    gal_pars["hlr_disk"] = Gal_Hlr
    gal_pars["mag_i"] = Gal_Mag
    
    psf_pars = {}
    psf_pars['randint'] = PSF_randint
    
    return gal_pars, psf_pars


# Get data loader
nSims = 20000
GalCat, PSFCat = dataframe(nSims, seed=12345)
train_ds = ShapeDataset(GalCat, PSFCat)
    
    
# Train the network
tr = train.Train()
tr.run(train_ds, show_log=True)