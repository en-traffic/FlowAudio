import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule

# import multiprocessing
# import datasets
# import pandas as pd
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# import pyarrow.parquet as pq

from ml.dataset import dataset_collate_function

class CNN1D(LightningModule):
    def __init__(self, data_path, output_dim):
        super().__init__()

        self.save_hyperparameters()

        self.conv1 = nn.Conv1d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 7, 256) #25_10
        self.fc2 = nn.Linear(256, self.hparams.output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def train_dataloader(self):
        data = torch.load(self.hparams.data_path, weights_only=True)
        X_train = data['X_train']
        y_train = data['y_train']
        X_train_tensor = torch.stack(X_train)
        y_train_tensor = torch.tensor(y_train)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, num_workers=16, shuffle=True)
        return train_loader

    def val_dataloader(self):
        data = torch.load(self.hparams.data_path, weights_only=True)
        X_train = data['X_val']
        y_train = data['y_val']
        X_train_tensor = torch.stack(X_train)
        y_train_tensor = torch.tensor(y_train)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, num_workers=16, shuffle=False)
        return train_loader


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)


    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("training_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss
