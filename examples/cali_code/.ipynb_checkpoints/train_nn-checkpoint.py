import numpy as np
from astropy.io import fits
import tenbilac

def train_NN_cali(file, model_path='./cali_wd/'):
    
    with fits.open(file) as hdul:
        raw_data = hdul[0].data 
    
    # Should rewrite this part
    case_size = 5000
    real_size = 2000

    # Get proper format
    inputs_ = raw_data[:,0:7].reshape(case_size,real_size,7)
    inputs = np.zeros((real_size, 4, case_size))
    labels = np.zeros((1,case_size))
    for i in range(case_size):
        inputs[:,:,i] = inputs_[i,:,1:5]
        labels[:,i] = inputs_[i,1000,5]/0.1 # 5 for g1, 6 for g2

    # Train
    ten = tenbilac.com.Tenbilac() # a folder and a config file should first be created
    ten.train(inputs, labels, auxinputs=np.array([None,None]))
    
    return


def train_NN_weight(file, model_path='./weight_wd/'):

    with fits.open(file) as hdul:
        inputs = hdul[0].data
        labels = hdul[1].data
        auxinputs = hdul[2].data
    
    # Train
    ten = tenbilac.com.Tenbilac(model_path) # a folder and a config file should first be created
    ten.train(inputs, labels, auxinputs=auxinputs)
    
    return


    
    