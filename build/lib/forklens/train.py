import numpy as np
from astropy.io import fits

import torch
from torch import optim, nn
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from .networks import ForkCNN, CaliNN

import sys,time,os
from .config_loader import Config


class BaseTrain(object):
    """
    Base class for training models.
    
    Handles common training tasks such as model loading, training loops,
    validation loops, and saving checkpoints.
    """
    
    def __init__(self, config, section='train'):
        """
        Initialize the BaseTrain object.

        Args:
            config (Config): Configuration object.
            section (str): The configuration section to use ('train' or 'nntrain').
        """
        self.config = config
        self.cfg = getattr(self.config, section)
        self.workers = self.cfg.workers
        self.device = torch.device(self.cfg.device)
        self.nGPUs = self.cfg.gpu_number
        
    def _set_data(self, dataset):
        """
        Set up the data loaders.
        
        Args:
            dataset (Dataset): The dataset to use.
        """
        raise NotImplementedError
        
    def _init_model(self):
        """
        Initialize the model architecture.
        
        Returns:
            nn.Module: The model.
        """
        raise NotImplementedError
        
    def load_model(self, path=None, strict=True):
        """
        Load the model, optionally from a checkpoint.

        Args:
            path (str, optional): Path to the checkpoint file.
            strict (bool, optional): Whether to strictly enforce state dict keys.

        Returns:
            nn.Module: The loaded model.
        """
        model = self._init_model()
        model.to(self.device)
        
        if path is not None:
            state_dict = torch.load(path)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=strict)

        if self.nGPUs > 1:
            model = nn.DataParallel(model, device_ids=range(self.nGPUs))
        
        return model
    
    def _init_criterion(self):
        """
        Initialize the loss function.
        
        Returns:
            nn.Module: The loss function.
        """
        raise NotImplementedError
        
    def _init_optimizer(self):
        """
        Initialize the optimizer.
        
        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        raise NotImplementedError
        
    def _step_scheduler(self, scheduler, train_loss, valid_loss):
        """
        Step the learning rate scheduler.

        Args:
            scheduler (torch.optim.lr_scheduler): The scheduler.
            train_loss (float): The training loss.
            valid_loss (float): The validation loss.
        """
        scheduler.step(train_loss)

    
    def _predict_step(self, batch, model, criterion):
        """
        Process a single batch for prediction.
        
        Args:
            batch (dict): The batch of data.
            model (nn.Module): The model to use.
            criterion (nn.Module): The loss function.
            
        Returns:
            dict: Dictionary containing 'preds', 'targets', 'loss', and optional 'ids', 'snr'.
        """
        raise NotImplementedError

    def _predict_generator(self, test_dl, model, criterion):
        """
        Generator for chunk-based prediction.
        """
        model.eval()
        
        if hasattr(self, '_prepare_predict'):
            self._prepare_predict(test_dl)
            
        with torch.no_grad():
            for i, batch in enumerate(test_dl):
                batch_out = self._predict_step(batch, model, criterion)
                yield batch_out

    def predict(self, test_dl, model=None, criterion=None, yield_chunks=False):
        """
        General prediction function.
        
        Args:
            test_dl (DataLoader): Test data loader.
            model (nn.Module, optional): Model to use. Defaults to self.model.
            criterion (nn.Module, optional): Loss function. Defaults to self.criterion or init new one.
            yield_chunks (bool, optional): If True, returns a generator yielding results per batch. 
                                           Useful for large datasets to save memory. Defaults to False.
            
        Returns:
            dict or generator: Dictionary with keys 'preds', 'targets', 'loss', and optionally 'ids', 'snr'.
                               If yield_chunks is True, yields dictionaries for each batch.
        """
        if model is None:
            model = self.model
            
        if criterion is None:
            if hasattr(self, 'criterion'):
                criterion = self.criterion
            else:
                try: 
                    criterion = self._init_criterion()
                except:
                    criterion = nn.MSELoss() # Fallback
        
        if yield_chunks:
            return self._predict_generator(test_dl, model, criterion)
            
        model.eval()
        device = self.device
        
        results = {
            'preds': [],
            'targets': [],
            'ids': [],
            'snr': [],
            'info': []
        }
        losses = []
        
        if hasattr(self, '_prepare_predict'):
            self._prepare_predict(test_dl)
        
        # Use simple enumerate as requested by user (matching their working _predictFunc pattern)
        with torch.no_grad():
            for i, batch in enumerate(test_dl):
                batch_out = self._predict_step(batch, model, criterion)
                
                losses.append(batch_out['loss'])
                
                # Accumulate results
                # We append validation batch results to list then concatenate at the end
                # accessible validation loop style
                results['preds'].append(batch_out['preds'])
                results['targets'].append(batch_out['targets'])
                
                if 'ids' in batch_out and batch_out['ids'] is not None:
                    results['ids'].append(batch_out['ids'])
                if 'snr' in batch_out and batch_out['snr'] is not None:
                    results['snr'].append(batch_out['snr'])
                if 'info' in batch_out and batch_out['info'] is not None:
                    results['info'].append(batch_out['info'])

        final_results = {}
        if len(losses) > 0:
            final_results['loss'] = np.sqrt(np.mean(losses))
        else:
            final_results['loss'] = 0.0
            
        if len(results['preds']) > 0:
            final_results['preds'] = np.concatenate(results['preds'], axis=0)
            final_results['targets'] = np.concatenate(results['targets'], axis=0)
        
        if len(results['ids']) > 0:
            final_results['ids'] = np.concatenate(results['ids'], axis=0)
        if len(results['snr']) > 0:
            final_results['snr'] = np.concatenate(results['snr'], axis=0)
        if len(results['info']) > 0:
            final_results['info'] = np.concatenate(results['info'], axis=0)
            
        return final_results

    # def predict(self, test_dl, criterion=nn.MSELoss()):

    #     self.model.eval()
    #     losses=[]
    #     for i, batch in enumerate(test_dl):
    #         inputs1, inputs2 = batch['gal_image'].float().to(self.device), \
    #                                    batch['psf_image'].float().to(self.device)
    #         outputs = self.model.forward(inputs1, inputs2)
    #         labels_true_batch = batch['label'].float().view(-1,self.features).to(self.device)
    #         loss = criterion(outputs, labels_true_batch)
    #         losses.append(loss.item())
    #         if i == 0:
    #             ids = batch['id'].numpy()
    #             labels = outputs.detach().cpu().numpy()
    #             labels_true = labels_true_batch.cpu()
    #             # snr = batch['snr'].numpy()
    #         else:
    #             ids = np.concatenate((ids, batch['id'].numpy()))
    #             labels = np.vstack((labels, outputs.detach().cpu().numpy()))
    #             labels_true = np.vstack((labels_true, labels_true_batch.cpu()))  
    #             # snr = np.concatenate((snr, batch['snr'].numpy()))

    #     combined_pred = np.column_stack((ids, labels))
    #     combined_true = np.column_stack((ids, labels_true))
    #     # combined_snr = np.column_stack((ids, snr))

    #     epoch_loss = np.sqrt(sum(losses) / len(losses))
    #     return combined_pred, combined_true, epoch_loss

    def _trainFunc(self, epoch, show_log=True):
        """
        Run one epoch of training.

        Args:
            epoch (int): The current epoch number.
            show_log (bool): Whether to print logs.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.train_dl):
            self.optimizer.zero_grad()
            loss = self._process_batch(batch)
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

    def _validFunc(self, epoch, show_log=True):
        """
        Run one epoch of validation.

        Args:
            epoch (int): The current epoch number.
            show_log (bool): Whether to print logs.

        Returns:
            float: The average validation loss for the epoch.
        """
        self.model.eval()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.valid_dl):
            loss = self._process_batch(batch)
            losses.append(loss.item())

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        epoch_time = time.time() - epoch_start
        if show_log:
            print("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                        epoch_time // 60, 
                                                                        epoch_time % 60))
        return epoch_loss

    def run(self, dataset, show_log=True):
        """
        Run the full training process.

        Args:
            dataset (Dataset): The dataset to use.
            show_log (bool): Whether to print logs.

        Returns:
            tuple: A tuple containing lists of training and validation losses.
        """
        self._set_data(dataset)
        
        self.model = self.load_model()
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        print('Begin training ...')
        
        train_losses = []
        valid_losses = []
        for epoch in range(self.cfg.epoch_number):
            train_loss = self._trainFunc(epoch, show_log=show_log)
            valid_loss = self._validFunc(epoch, show_log=show_log)
            self._step_scheduler(scheduler, train_loss, valid_loss)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            if self.cfg.save_model:
                if not os.path.exists(self.cfg.model_path):
                    os.makedirs(self.cfg.model_path)
                torch.save(self.model.state_dict(), 
                           self.cfg.model_path+self.cfg.model_name+str(epoch))
                
        if self.cfg.save_model:
            hdu0 = fits.PrimaryHDU(train_losses)
            hdu1 = fits.ImageHDU(valid_losses)
            hdul = fits.HDUList([hdu0, hdu1])
            hdul.writeto(self.cfg.model_path+'/training_loss.fits', overwrite=True)
                
        print('Finish training !')
        return train_losses, valid_losses


class Train(BaseTrain):
    """
    Trainer for the main ForkCNN model.
    """
    
    def __init__(self, config):
        """
        Initialize the Train object.

        Args:
            config (Config): Configuration object.
        """
        super().__init__(config, 'train')
        self.batch_size = self.cfg.batch_size
        self.features = self.cfg.feature_number
        
    def _set_data(self, train_ds):
        '''
        Spilt the dataset into training and validation, and build dataloader.
        '''
        

        
        valid_split = self.cfg.validation_split

        size = len(train_ds)
        indices = list(range(size))
        split = int(np.floor(valid_split * size))

        train_indices, valid_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        self.train_dl = DataLoader(train_ds, 
                              batch_size=self.batch_size, 
                              num_workers=self.workers,
                              sampler=train_sampler,
                              pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True if self.workers > 0 else False)
        self.valid_dl = DataLoader(train_ds, 
                              batch_size=self.batch_size, 
                              num_workers=self.workers,
                              sampler=valid_sampler,
                              pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True if self.workers > 0 else False)
        print("Train_dl: {} Validation_dl: {}".format(len(self.train_dl), len(self.valid_dl)))
        
    def _init_model(self):
        gal_shape = (self.config.simulation.galaxy_stamp_size, self.config.simulation.galaxy_stamp_size)
        psf_shape = (self.config.simulation.psf_stamp_size, self.config.simulation.psf_stamp_size)
        return ForkCNN(self.features, self.batch_size, self.nGPUs, gal_shape=gal_shape, psf_shape=psf_shape)
    
    def _init_criterion(self):
        return nn.MSELoss()
    
    def _init_optimizer(self):
        return optim.SGD(self.model.parameters(), 
                         lr=self.cfg.initial_learning_rate, 
                         momentum=self.cfg.momentum)

    def _process_batch(self, batch):
        inputs1, inputs2, labels = batch['gal_image'].float().to(self.device), \
                                   batch['psf_image'].float().to(self.device), \
                                   batch['label'].float().view(-1,self.features).to(self.device)
        outputs = self.model.forward(inputs1, inputs2)
        loss = self.criterion(outputs, labels)
        return loss

    def _predict_step(self, batch, model, criterion):
        inputs1, inputs2 = batch['gal_image'].float().to(self.device), \
                           batch['psf_image'].float().to(self.device)
        labels = batch['label'].float().view(-1,self.features).to(self.device)
        
        outputs = model(inputs1, inputs2)
        loss = criterion(outputs, labels)
        
        out = {
            'loss': loss.item(),
            'preds': outputs.cpu().numpy(),
            'targets': labels.cpu().numpy(),
            'ids': batch['id'].numpy() if 'id' in batch else None,
            'snr': batch['snr'].numpy() if 'snr' in batch else None,
            'info': batch['info'].numpy() if 'info' in batch else None
        }
        return out
    
    
###############################################
###
    
    
class MSBLoss(nn.Module):
    """
    Mean Squared Bias Loss.
    
    Calculates the mean squared error between the mean of predictions and the mean of targets.
    """
    def __init__(self):
        super(MSBLoss, self).__init__()
        
    def forward(self,x,y):
        """
        Calculate the loss.

        Args:
            x (torch.Tensor): Predictions.
            y (torch.Tensor): Targets.

        Returns:
            torch.Tensor: The calculated loss.
        """
        
        if torch.std(y,axis=1).any() != 0:
            print('Waring!')
            # print(y)
        
        # print(x.shape)
        # print(y.shape)
        l = torch.mean(((torch.mean(x,axis=1)-torch.mean(y,axis=1))**2),axis=0)
        # print(torch.mean(x,axis=1)-torch.mean(y,axis=1))
        # print(l)
        return l
    
    
class NNTrain(BaseTrain):
    """
    Trainer for the calibration Neural Network.
    """
    
    def __init__(self, config):
        """
        Initialize the NNTrain object.

        Args:
            config (Config): Configuration object.
        """
        super().__init__(config, 'nntrain')
        self.batch_cases = self.cfg.batch_cases
        
    def run(self, dataset, show_log=True):
        """
        Run the training process for NNTrain.
        
        Args:
            dataset (Dataset): The dataset to use.
            show_log (bool): Whether to print logs.

        Returns:
            tuple: A tuple containing lists of training and validation losses.
        """
        self.real_size = getattr(dataset, "real_size", 1)
        return super().run(dataset, show_log)

    def _set_data(self, train_ds):
        '''
        Spilt the dataset into training and validation, and build dataloader.
        '''
        valid_split = self.cfg.validation_split

        size = len(train_ds)
        indices = list(range(size))
        split = int(np.floor(valid_split * size))

        train_indices, valid_indices = indices[split:], indices[:split]

        if self.real_size > 1:
             bs = self.real_size*self.batch_cases
        else:
             bs = self.cfg.batch_size

        self.train_dl = DataLoader(Subset(train_ds, train_indices), 
                              batch_size=bs, 
                              num_workers=self.workers,
                              pin_memory=True,
                              prefetch_factor=2 if self.workers > 0 else None,
                              persistent_workers=True if self.workers > 0 else False)
        self.valid_dl = DataLoader(Subset(train_ds, valid_indices), 
                              batch_size=bs,
                              num_workers=self.workers,
                              pin_memory=True,
                              prefetch_factor=2 if self.workers > 0 else None,
                              persistent_workers=True if self.workers > 0 else False)
        print("Train_dl: {} Validation_dl: {}".format(len(self.train_dl), len(self.valid_dl)))
        
    def _init_model(self):
        return CaliNN(in_features=self.cfg.in_features,
                      out_features=self.cfg.out_features,
                      hidden_features=self.cfg.hidden_features)
    
    def _init_criterion(self):
        return nn.MSELoss()
    
    def _init_optimizer(self):
        return optim.Adam(self.model.parameters(), 
                          lr=self.cfg.initial_learning_rate,
                          betas=self.cfg.adam_betas)
    
    def _step_scheduler(self, scheduler, train_loss, valid_loss):
        scheduler.step(valid_loss)

    def _process_batch(self, batch):
        inputs, labels = batch['input'].float().to(self.device), \
                         batch['label'].float().to(self.device)
        outputs = self.model.forward(inputs)
        
        if self.real_size > 1:
             # remember to reshape it before feeding into loss calculation
             outputs = torch.reshape(outputs,(-1,self.real_size))
             labels = torch.reshape(labels,(-1,self.real_size))
             # For legacy calibration with multiple realizations, one might want MSBLoss here,
             # but user specifically asked for MSE. 
             # Also nn.MSELoss handles standard broadcasting if shapes match.
             # If reshaping is done, outputs and labels are (batch_cases, real_size).
             
        loss = self.criterion(outputs, labels) 
        return loss

    def _prepare_predict(self, test_dl):
        # Ensure real_size is correct for the test dataset
        self.real_size = getattr(test_dl.dataset, "real_size", 1)

    def _predict_step(self, batch, model, criterion):
        inputs = batch['input'].float().to(self.device)
        labels = batch['label'].float().to(self.device)
        outputs = model(inputs)
        
        real_size = getattr(self, 'real_size', 1)
        
        if real_size > 1:
             outputs = torch.reshape(outputs,(-1, real_size))
             labels = torch.reshape(labels,(-1, real_size))
        
        loss = criterion(outputs, labels)
        
        if real_size > 1:
             # Return mean over realizations
             preds = torch.mean(outputs, dim=1)
             targets = torch.mean(labels, dim=1)
        else:
             preds = outputs
             targets = labels
             
        out = {
            'loss': loss.item(),
            'preds': preds.cpu().numpy(),
            'targets': targets.cpu().numpy()
        }
        return out

    
def cali_predict(test_dl,MODEL,config,criterion=nn.MSELoss()):
    """
    Run prediction for calibration.

    Args:
        test_dl (DataLoader): Test data loader.
        MODEL (nn.Module): The model to use for prediction.
        config (Config): Configuration object.
        criterion (nn.Module): Loss function.

    Returns:
        tuple: Predicted residuals, true labels, and average loss.
    """

    MODEL.eval()
    losses=[]
    real_size = getattr(test_dl.dataset, 'real_size', 1)

    for i, batch in enumerate(test_dl):
        inputs, labels = batch['input'].float().to(torch.device(config.nntrain.device)), \
                         batch['label'].float().to(torch.device(config.nntrain.device))
        outputs = MODEL.forward(inputs)
        
        if real_size > 1:
            outputs = torch.reshape(outputs,(-1, real_size))
            labels = torch.reshape(labels,(-1, real_size))
            
            res_batch = np.mean(outputs.detach().cpu().numpy(),axis=1)
            labels_batch = np.mean(labels.cpu().numpy(),axis=1)
        else:
             res_batch = outputs.detach().cpu().numpy()
             labels_batch = labels.cpu().numpy()

        loss = criterion(outputs, labels)
        losses.append(loss.item())

        if i == 0:
            res = res_batch
            labels_true = labels_batch
        else:
            res = np.concatenate((res, res_batch),axis=0)
            labels_true = np.concatenate((labels_true, labels_batch),axis=0)  

    epoch_loss = np.sqrt(sum(losses) / len(losses))
    return res, labels_true, epoch_loss
