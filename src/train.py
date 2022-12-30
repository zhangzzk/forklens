import numpy as np
from astropy.io import fits

import torch
from torch import optim, nn
from torch.utils.data import SubsetRandomSampler, DataLoader

from .cnn import ForkCNN

import sys,time,os
# sys.path.append(os.path.abspath("../configs"))
# from configs import config
import config

class Train(object):
    
    def __init__(self):
        
        # Device Options   
        self.workers = config.train['workers']
        self.device = torch.device(config.train['device'])
        self.batch_size = config.train['batch_size']
        self.features = config.train['feature_number']
        self.nGPUs = config.train['gpu_number']
        
        
    def _set_data(self, train_ds):
        '''
        Spilt the dataset into training and validation, and build dataloader.
        '''
        
        valid_split = config.train['validation_split']

        size = len(train_ds)
        indices = list(range(size))
        split = int(np.floor(valid_split * size))

        train_indices, valid_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        self.train_dl = DataLoader(train_ds, 
                              batch_size=self.batch_size, 
                              num_workers=self.workers,
                              sampler=train_sampler)
        self.valid_dl = DataLoader(train_ds, 
                              batch_size=self.batch_size, 
                              num_workers=self.workers,
                              sampler=valid_sampler)
        print("Train_dl: {} Validation_dl: {}".format(len(self.train_dl), len(self.valid_dl)))
        
    def load_model(self,path=None,strict=True):
        
        model = ForkCNN(self.features, self.batch_size, self.nGPUs)
        model.to(self.device)
        if self.nGPUs > 1:
            model = nn.DataParallel(model, device_ids=range(self.nGPUs))
        
        if path != None:
            model.load_state_dict(torch.load(path), strict=strict)
        
        return model
    
        
        
    def run(self, dataset, show_log=True):
        
        # set data loader here
        self._set_data(dataset)
        
        # self.model = ForkCNN(self.features, self.batch_size, self.nGPUs)
        # if self.nGPUs > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=range(self.nGPUs))
        # self.model.to(self.device)
        
        self.model = self.load_model()
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), 
                                   lr=config.train['initial_learning_rate'], 
                                   momentum=config.train['momentum'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        
        print('Begin training ...')
        
        # Loop the training and validation processes
        train_losses = []
        valid_losses = []
        for epoch in range(config.train['epoch_number']):
            train_loss = self._trainFunc(epoch,show_log=show_log)
            valid_loss = self._validFunc(epoch,show_log=show_log)
            scheduler.step(train_loss)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            if config.train['save_model']:
                if not os.path.exists(config.train['model_path']):
                    os.makedirs(config.train['model_path'])
                torch.save(self.model.state_dict(), 
                           config.train['model_path']+config.train['model_name']+str(epoch))
                
        hdu0 = fits.PrimaryHDU(train_losses)
        hdu1 = fits.ImageHDU(valid_losses)
        hdul = fits.HDUList([hdu0, hdu1])
        hdul.writeto(config.train['model_path']+'/training_loss.fits',overwrite=True)
                
        print('Finish training !')
        return train_losses, valid_losses
        

    def _trainFunc(self,epoch,show_log=True):
        self.model.train()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.train_dl):
            inputs1, inputs2, labels = batch['gal_image'].float().to(self.device), \
                                       batch['psf_image'].float().to(self.device), \
                                       batch['label'].float().view(-1,self.features).to(self.device)

            self.optimizer.zero_grad()             
            outputs = self.model.forward(inputs1, inputs2)
            loss = self.criterion(outputs, labels) 
            losses.append(loss.item())        
            loss = torch.sqrt(loss)           
            loss.backward()                   
            self.optimizer.step()                  

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        epoch_time = time.time() - epoch_start
        if show_log:
            print("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                          epoch_time // 60, 
                                                                          epoch_time % 60))
        return epoch_loss

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.valid_dl):
            inputs1, inputs2, labels = batch['gal_image'].float().to(self.device), \
                                       batch['psf_image'].float().to(self.device), \
                                       batch['label'].float().view(-1,self.features).to(self.device)

            outputs = self.model.forward(inputs1, inputs2)
            loss = self.criterion(outputs, labels)
            losses.append(loss.item())

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        epoch_time = time.time() - epoch_start
        if show_log:
            print("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                        epoch_time // 60, 
                                                                        epoch_time % 60))
        return epoch_loss
    

    def _predictFunc(self,test_dl,MODEL,criterion=nn.MSELoss()):

        MODEL.eval()
        losses=[]
        for i, batch in enumerate(test_dl):
            inputs1, inputs2 = batch['gal_image'].float().to(self.device), \
                                       batch['psf_image'].float().to(self.device)
            outputs = MODEL.forward(inputs1, inputs2)
            labels_true_batch = batch['label'].float().view(-1,self.features).to(self.device)
            loss = criterion(outputs, labels_true_batch)
            losses.append(loss.item())
            if i == 0:
                ids = batch['id'].numpy()
                labels = outputs.detach().cpu().numpy()
                labels_true = labels_true_batch.cpu()
                snr = batch['snr'].numpy()
            else:
                ids = np.concatenate((ids, batch['id'].numpy()))
                labels = np.vstack((labels, outputs.detach().cpu().numpy()))
                labels_true = np.vstack((labels_true, labels_true_batch.cpu()))  
                snr = np.concatenate((snr, batch['snr'].numpy()))

        combined_pred = np.column_stack((ids, labels))
        combined_true = np.column_stack((ids, labels_true))
        combined_snr = np.column_stack((ids, snr))

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        return combined_pred, combined_true, combined_snr, epoch_loss