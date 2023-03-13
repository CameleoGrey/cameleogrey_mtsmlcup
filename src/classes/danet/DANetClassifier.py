
import numpy as np
from copy import deepcopy
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam, AdamW
from qhoptim.pyt import QHAdam

from classes.danet.DANet import DANet


class CustomDataset(Dataset):
    def __init__(self, x, y, class_count):
        
        self.x = x.astype( np.float32 )
        self.y = y.astype( np.float32 )
        self.ds_len = len(y)
        self.class_count = class_count
        
        y_transformed = np.zeros((self.ds_len, self.class_count), dtype=np.float32)
        for i in range(len(y)):
            y_transformed[i][y[i]] = 1.0
        self.y = y_transformed
    
        pass
    
    def __getitem__(self, id):
        
        x = self.x[id]
        y = self.y[id]
        
        return x, y
    
    def __len__(self):
        return self.ds_len
    
class PredictDataset(Dataset):
    def __init__(self, x):
        self.x = x.astype( np.float32 )
        self.ds_len = len(x)
        pass
    
    def __getitem__(self, id):
        x = self.x[id]
        return x
    
    def __len__(self):
        return self.ds_len 
        
        

class DANetClassifier():
    def __init__(self, input_dim, num_classes, 
                 #layer_num=48, base_outdim=96, k=8,
                 layer_num=32, base_outdim=64, k=5,
                 virtual_batch_size=256, drop_rate=0.1,
                 device="cuda"):
        
        self.device = device
        
        self.danet = DANet(input_dim = input_dim, 
                           num_classes = num_classes, 
                           layer_num = layer_num, 
                           base_outdim = base_outdim, 
                           k = k, 
                           virtual_batch_size = virtual_batch_size, 
                           drop_rate = drop_rate)
        self.model = torch.nn.Sequential( self.danet, nn.LogSoftmax(dim=1) )
        self.model = self.model.to( self.device )
        
        self.class_names = None
        
        pass
    
    def predict_proba(self, x, batch_size=1024):
        
        self.model.eval()
        
        predict_dataset = PredictDataset( x )
        predict_dataloader = DataLoader( predict_dataset, batch_size=batch_size, shuffle=False )
        
        probas = []
        for x_batch in predict_dataloader:
            x_batch = x_batch.to( self.device )
            y_pred = self.model( x_batch )
            y_pred = nn.Softmax(dim=1)(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            probas.append( y_pred )
        probas = np.vstack( probas )
        
        return probas
    
    def predict(self, x, batch_size=1024):
        
        probas = self.predict_proba(x, batch_size)
        
        y_pred = []
        for i in range(len(probas)):
            current_proba = probas[i]
            y_i = np.argmax( current_proba )
            y_i = self.class_names[ y_i ]
            y_pred.append( y_i )
        y_pred = np.array( y_pred ) 
            
        return y_pred
    
    def get_embeddings(self, x, batch_size=1024):
        pass
    
    # no mixup version
    """def fit(self, x_train, y_train, x_val, y_val,
            start_lr=0.008, end_lr=0.0001, batch_size=8192, epochs=100):
        
        def train(dataloader, loss_fn, optimizer):
            self.model.train()

            i = 0
            start_time = datetime.now()
            ema_loss = None
            log_frequency = int(len(dataloader) / 10.0)
            for x, y in dataloader:
                x = x.to( self.device )
                y = y.to( self.device )
                y_pred = self.model(x)

                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i += 1
                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    alpha = 0.2
                    ema_loss = alpha * loss.item() + (1.0 - alpha) * ema_loss

                if i % log_frequency == 0:
                    total_time = datetime.now() - start_time
                    print(f"ema_loss: {ema_loss:>7f}  [{i * len(x):>5d}/{len(dataloader.dataset):>5d}] {total_time}")

        def test(dataloader, loss_fn):
            self.model.eval()
            
            num_batches = len(dataloader)
            test_loss = 0
            with torch.no_grad():
                for x, y in dataloader:
                    x = x.to( self.device )
                    y = y.to( self.device )
                    y_pred = self.model(x)
                    test_loss += loss_fn(y_pred, y).item()
            test_loss /= num_batches
            return test_loss
        
        self.class_names = np.unique( y_train )
        class_count = len( self.class_names )
        train_dataset = CustomDataset( x_train, y_train, class_count )
        val_dataset = CustomDataset( x_val, y_val, class_count )
        val_data_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        loss_function = torch.nn.CrossEntropyLoss()
        
        best_loss = np.inf
        lr_step = (end_lr - start_lr) / epochs
        current_lr = start_lr
        for i in range(1, epochs+1):
            
            #optimizer = AdamW(self.model.parameters(), lr=current_lr, weight_decay=1e-2, betas=(0.9, 0.999), amsgrad=True)
            #optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr, weight_decay=1e-5, betas=(0.9, 0.999), amsgrad=True)
            optimizer = QHAdam(self.model.parameters(), lr=current_lr, weight_decay=1.0e-5 )
            
            print("Epoch: {} | lr: {}".format(i, current_lr))
            train(train_data_loader, loss_function, optimizer)
            
            val_loss = test(val_data_loader, loss_function)
            print("Validation loss: {}".format(val_loss))
            if val_loss < best_loss:
                print("Previous best loss: {}".format(best_loss))
                best_loss = val_loss
                best_model = deepcopy( self.model )
            
            current_lr += lr_step
        
        self.model = best_model
        self.model.eval()
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        return self"""
    
    #mixup version
    def fit(self, x_train, y_train, x_val, y_val,
            start_lr=0.008, end_lr=0.0001, batch_size=8192, epochs=1000):
        
        def train(dataloader_1, dataloader_2, loss_fn, optimizer):
            self.model.train()

            i = 0
            start_time = datetime.now()
            ema_loss = None
            log_frequency = int(len(dataloader_1) / 10.0)
            for (x_1, y_1), (x_2, y_2) in zip(dataloader_1, dataloader_2):
                lam = np.random.beta(0.2, 0.2)
                x = lam * x_1 + (1.0 - lam) * x_2
                y = lam * y_1 + (1.0 - lam) * y_2
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)

                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i += 1
                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    alpha = 0.2
                    ema_loss = alpha * loss.item() + (1.0 - alpha) * ema_loss

                #if i % log_frequency == 0:
                total_time = datetime.now() - start_time
                print(f"ema_loss: {ema_loss:>7f}  [{i * len(x):>5d}/{len(dataloader_1.dataset):>5d}] {total_time}")

        def test(dataloader, loss_fn):
            self.model.eval()
            
            num_batches = len(dataloader)
            test_loss = 0
            with torch.no_grad():
                for x, y in dataloader:
                    x = x.to( self.device )
                    y = y.to( self.device )
                    y_pred = self.model(x)
                    test_loss += loss_fn(y_pred, y).item()
            test_loss /= num_batches
            return test_loss
        
        self.class_names = np.unique( y_train )
        class_count = len( self.class_names )
        train_dataset_1 = CustomDataset( x_train, y_train, class_count )
        train_dataset_2 = CustomDataset( x_train, y_train, class_count )
        train_data_loader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True, drop_last=True)
        train_data_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataset = CustomDataset( x_val, y_val, class_count )
        val_data_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        loss_function = torch.nn.CrossEntropyLoss()
        
        best_loss = np.inf
        lr_step = (end_lr - start_lr) / epochs
        current_lr = start_lr
        for i in range(1, epochs+1):
            
            #optimizer = AdamW(self.model.parameters(), lr=current_lr, weight_decay=1e-2, betas=(0.9, 0.999), amsgrad=True)
            #optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr, weight_decay=1e-5, betas=(0.9, 0.999), amsgrad=True)
            #if i % 20 == 0:
            #    current_lr = 0.95 * current_lr
            optimizer = QHAdam( self.model.parameters(), lr=current_lr, weight_decay=1.0e-5 )
            
            print("Epoch: {} | lr: {}".format(i, current_lr))
            train(train_data_loader_1, train_data_loader_2, loss_function, optimizer)
            
            val_loss = test(val_data_loader, loss_function)
            print("Validation loss: {}".format(val_loss))
            if val_loss < best_loss:
                print("Previous best loss: {}".format(best_loss))
                best_loss = val_loss
                best_model = deepcopy( self.model )
            
            current_lr += lr_step
        
        self.model = best_model
        
        self.model = best_model
        self.model.eval()
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        return self
    