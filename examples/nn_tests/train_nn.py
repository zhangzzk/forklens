import config
from forklens import train
from forklens.dataset import NNDataset
from torch.utils.data import DataLoader

import numpy as np
# import matplotlib.pyplot as plt
from astropy.io import fits


# Get measured catalog
with fits.open('../cnn_tests/TypeI_dataset_5000case_2000real_-0.1-0.1shear.fits') as hdul:
    raw_data = hdul[0].data
    
Dataframe = {}
Dataframe['prediction'] = raw_data[:,1:5]
Dataframe['true_shear'] = raw_data[:,5] # g1
train_ds = NNDataset(Dataframe)
    
tr = train.NNTrain()
tr.run(train_ds)