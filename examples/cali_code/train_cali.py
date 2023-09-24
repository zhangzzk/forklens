import numpy as np
from astropy.io import fits
import tenbilac
from time import time

# Get measured catalog
file = './TypeI_5000case_2000real.fits'
print('Read file: ',file)
with fits.open(file) as hdul:
    raw_data = hdul[0].data 
    
case_size = 5000
real_size = 2000

# re-normalize
# raw_data[]
    
# Get proper format
inputs_ = raw_data[:,0:7].reshape(case_size,real_size,7)
inputs = np.zeros((real_size, 4, case_size))
labels = np.zeros((1,case_size))
for i in range(case_size):
    inputs[:,:,i] = inputs_[i,:,1:5]
    labels[:,i] = inputs_[i,1000,5]/0.1 # 5 for g1, 6 for g2
print(inputs.shape)
print(labels.shape)


# Train the data
print('Start training ... ')
start = time()
ten = tenbilac.com.Tenbilac('./cali_wd/') # a folder and a config file should first be created
ten.train(inputs, labels, auxinputs=np.array([None,None]))
print('Done training!')
print('Time consumed: %.2f mins.'%((time()-start)/60))
