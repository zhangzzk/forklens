import numpy as np
from astropy.io import fits
import tenbilac
from time import time

# Get measured catalog
file = './TypeII_200_100000.fits'
print('Read file: ',file)
with fits.open(file) as hdul:
    inputs = hdul[0].data
    labels = hdul[1].data
    auxinputs = hdul[2].data
print(inputs.shape)
print(labels.shape)
print(auxinputs.shape)


# Train the data
print('Start training ... ')
start = time()
ten = tenbilac.com.Tenbilac('./weight_wd/') # a folder and a config file should first be created
ten.train(inputs, labels, auxinputs=auxinputs)
print('Done training!')
print('Time consumed: %.2f mins.'%((time()-start)/60))
