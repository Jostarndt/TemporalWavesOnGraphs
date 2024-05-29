import torch
import numpy as np

import sys
import os
import pandas as pd
import xarray as xr

import pdb
import datetime


class SimulationDataset(torch.utils.data.Dataset):
    path = os.path.join('..', 'data', 'SI_pde', '364_25_numpy_dataset.npy')

    def __init__(self, context_len, forecast_len, stride=1, transform = None):
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.stride = stride
        self.data = torch.tensor(np.load(self.path)[:,:,1], dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return (len(self.data) - 2*(self.context_len + self.forecast_len)) // self.stride # 2 because training and test-set

    def __getitem__(self, idx):
        context_data = self.data[idx*self.stride : idx*self.stride + self.context_len,:]
        target_data = self.data[idx*self.stride + self.context_len : idx*self.stride + self.context_len + self.forecast_len,:]
        if self.transform:
            context_data = self.transform(context_data)
            #target_data = self.transform(target_data)
        return context_data, target_data


class SimulationDenoisingDataset(torch.utils.data.Dataset):
    path = os.path.join('..', 'data', 'SI_pde', '364_25_numpy_dataset.npy')

    def __init__(self, context_len, forecast_len, stride=1, transform = None):
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.stride = stride
        self.data = torch.tensor(np.load(self.path)[:,:,1], dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return (len(self.data) - 2*(self.context_len + self.forecast_len)) // self.stride # 2 because training and test-set

    def __getitem__(self, idx):
        context_data = self.transform(self.data[idx*self.stride : idx*self.stride + self.context_len,:])
        target_data = self.data[idx*self.stride + self.context_len : idx*self.stride + self.context_len + self.forecast_len,:]

        return context_data, target_data     



class WaveDataset(torch.utils.data.Dataset):
    path = os.path.join('..', 'data', 'wave_equation', 'wave_dataset.npy')
    path = "/srv/data/csvs/simulation/wave_dataset.npy" #TODO fix


    def __init__(self, context_len, forecast_len, stride=1, transform = None):
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.stride = stride
        self.data = torch.tensor(np.load(self.path)[:,:,0], dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return (len(self.data) - 2*(self.context_len + self.forecast_len)) // self.stride # 2 because training and test-set

    def __getitem__(self, idx):
        context_data = self.data[idx*self.stride : idx*self.stride + self.context_len,:]
        target_data = self.data[idx*self.stride + self.context_len : idx*self.stride + self.context_len + self.forecast_len,:]
        if self.transform:
            context_data = self.transform(context_data)
            #target_data = self.transform(target_data)
        return context_data, target_data


class AdvectionDiffusionDataset(torch.utils.data.Dataset):
    path = os.path.join('..', 'data', 'advection_diffusion', 'advection_diffusion_numpy_dataset.npy')
    path = "/srv/data/csvs/simulation/advection_diffusion_numpy_dataset.npy" #TODO fix

    def __init__(self, context_len, forecast_len, stride=1, transform = None):
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.stride = stride
        self.data = torch.tensor(np.load(self.path)[:,:,0], dtype=torch.float32)
        self.data[self.data < 1e-8] = 0	
        self.transform = transform

    def __len__(self):
        return (len(self.data) - 2*(self.context_len + self.forecast_len)) // self.stride # 2 because training and test-set

    def __getitem__(self, idx):
        context_data = self.data[idx*self.stride : idx*self.stride + self.context_len,:]
        target_data = self.data[idx*self.stride + self.context_len : idx*self.stride + self.context_len + self.forecast_len,:]
        if self.transform:
            context_data = self.transform(context_data)
            #target_data = self.transform(target_data)
        return context_data, target_data                                                  