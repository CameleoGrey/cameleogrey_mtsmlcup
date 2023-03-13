
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn import MSELoss

from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

from classes.MLP_Network import MLP_Network

class AutoEncoderDataset(Dataset):
    def __init__(self, x):
        self.x = x
        pass
    
    def __len__(self):
        x_len = len(self.x)
        return x_len
    
    def __getitem__(self, index):
        current_x = self.x[ index ].astype(np.float32)
        return current_x
        
    

class GreyAutoencoder1d():
    def __init__(self, input_dim, embedding_dim=128, hidden_layer_dim=128, hidden_layers_num=3, dropout_rate=0.05, device="cuda"):
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_layer_dim = hidden_layer_dim,
        self.hidden_layers_num = hidden_layers_num
        self.dropout = dropout_rate
        self.device = device
        
        self.encoder = MLP_Network(input_dim = self.input_dim,
                                   output_dim = self.embedding_dim,
                                   hidden_layer_dim = self.hidden_layer_dim, 
                                   hidden_layers_num = self.hidden_layers_num, 
                                   dropout_rate = self.dropout)
        
        self.decoder = MLP_Network(input_dim = self.embedding_dim,
                                   output_dim = self.input_dim,
                                   hidden_layer_dim = self.hidden_layer_dim, 
                                   hidden_layers_num = self.hidden_layers_num, 
                                   dropout_rate = self.dropout)
        
        self.autoencoder = torch.nn.Sequential( self.encoder, self.decoder )
        
        self.encoder = self.encoder.float()
        self.decoder = self.decoder.float()
        self.autoencoder = self.autoencoder.float()
        
        self.encoder = self.encoder.to( self.device )
        self.decoder = self.decoder.to( self.device )
        self.autoencoder = self.autoencoder.to( self.device )
        
        pass
    
    
    # no mixup version
    """def fit(self, x_dataset, batch_size=1024, epochs=100, lr=0.001):
        
        def train(dataloader, model, loss_fn, optimizer):
            model.train()

            i = 0
            start_time = datetime.now()
            ema_loss = None
            for x in dataloader:
                x = x.to(self.device)
                pred = model(x)

                loss = loss_fn(pred, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i += 1
                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    alpha = 0.01
                    ema_loss = alpha * loss.item() + (1.0 - alpha) * ema_loss

                if i % 100 == 0:
                    total_time = datetime.now() - start_time
                    print(f"ema_loss: {ema_loss:>7f}  [{i * len(x):>5d}/{len(dataloader.dataset):>5d}] {total_time}")

        def test(dataloader, model, loss_fn):
            num_batches = len(dataloader)
            model.eval()
            test_loss = 0
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.no_grad():
                for x in dataloader:
                    x = x.to(device)
                    pred = model(x)
                    test_loss += loss_fn(pred, x).item()
            test_loss /= num_batches
            return test_loss

        train_dataset = AutoEncoderDataset( x_dataset )
        val_dataset = AutoEncoderDataset( x_dataset )
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        loss_function = torch.nn.MSELoss()
        
        
        x_dataset = x_dataset.copy()
        x_dataset = x_dataset.astype(np.float32)
        best_loss = np.inf
        for i in range(epochs):
            print("Epoch: {}".format(i))
            optimizer = AdamW(self.autoencoder.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.999), amsgrad=True)

            train(train_data_loader, self.autoencoder, loss_function, optimizer)
            val_loss = test(val_data_loader, self.autoencoder, loss_function)
            print("Validation loss: {}".format(val_loss))
            if val_loss < best_loss:
                print("Previous best loss: {}".format(best_loss))
                best_loss = val_loss
                best_encoder = deepcopy( self.encoder )
                best_decoder = deepcopy( self.decoder )
                best_autoencoder = deepcopy( self.autoencoder )
        
        self.encoder = best_encoder
        self.decoder = best_decoder
        self.autoencoder = best_autoencoder
        
        return self"""
                
    # mixup version
    def fit(self, x_dataset, batch_size=1024, epochs=100, lr=0.001):
        
        def train(dataloader_1, dataloader_2, model, loss_fn, optimizer):
            model.train()

            i = 0
            start_time = datetime.now()
            ema_loss = None
            for x_1, x_2 in zip(dataloader_1, dataloader_2):
                lam = np.random.beta(0.2, 0.2)
                x = lam * x_1 + (1.0 - lam) * x_2
                x = x.to(self.device)
                pred = model(x)

                loss = loss_fn(pred, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i += 1
                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    alpha = 0.01
                    ema_loss = alpha * loss.item() + (1.0 - alpha) * ema_loss

                if i % 100 == 0:
                    total_time = datetime.now() - start_time
                    print(f"ema_loss: {ema_loss:>7f}  [{i * len(x):>5d}/{len(dataloader_1.dataset):>5d}] {total_time}")

        def test(dataloader, model, loss_fn):
            num_batches = len(dataloader)
            model.eval()
            test_loss = 0
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.no_grad():
                for x in dataloader:
                    x = x.to(device)
                    pred = model(x)
                    test_loss += loss_fn(pred, x).item()
            test_loss /= num_batches
            return test_loss

        train_dataset_1 = AutoEncoderDataset( x_dataset )
        train_dataset_2 = AutoEncoderDataset( x_dataset )
        train_data_loader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True, drop_last=True)
        train_data_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataset = AutoEncoderDataset( x_dataset )
        val_data_loader = DataLoader(val_dataset, batch_size=10*batch_size, shuffle=False)
        loss_function = torch.nn.MSELoss()
        
        
        x_dataset = x_dataset.copy()
        x_dataset = x_dataset.astype(np.float32)
        best_loss = np.inf
        for i in range(epochs):
            print("Epoch: {}".format(i))
            #optimizer = AdamW(self.autoencoder.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.999), amsgrad=True)
            optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999), amsgrad=True)

            train(train_data_loader_1, train_data_loader_2, self.autoencoder, loss_function, optimizer)
            val_loss = test(val_data_loader, self.autoencoder, loss_function)
            print("Validation loss: {}".format(val_loss))
            if val_loss < best_loss:
                print("Previous best loss: {}".format(best_loss))
                best_loss = val_loss
                best_encoder = deepcopy( self.encoder )
                best_decoder = deepcopy( self.decoder )
                best_autoencoder = deepcopy( self.autoencoder )
        
        self.encoder = best_encoder
        self.decoder = best_decoder
        self.autoencoder = best_autoencoder
        
        return self
    
    def encode(self, x_dataset, batch_size=1024):
        
        self.encoder.eval()
        
        x_dataset = x_dataset.copy()
        x_dataset = x_dataset.astype(np.float32)
        
        x_embeddings = []
        test_dataset = AutoEncoderDataset( x_dataset )
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        for x_batch in test_data_loader:
            x_batch = x_batch.to(self.device)
            x_pred = self.encoder( x_batch )
            x_pred = x_pred.cpu().detach().numpy()
            x_embeddings.append( x_pred )
        x_embeddings = np.vstack( x_embeddings )
        
        return x_embeddings