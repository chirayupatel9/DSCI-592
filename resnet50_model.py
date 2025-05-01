"""
ResNet50-based model for material property prediction.
This module implements a ResNet50-based model for predicting material properties.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import os
import subprocess

def get_least_utilized_gpu():
    """Get the GPU with the least memory usage."""
    try:
        # Run nvidia-smi to get GPU memory usage
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                              capture_output=True, text=True)
        memory_usage = [int(x) for x in result.stdout.strip().split('\n')]
        
        # Find GPU with minimum memory usage
        min_usage = min(memory_usage)
        gpu_id = memory_usage.index(min_usage)
        
        print(f"Selected GPU {gpu_id} with {min_usage}MB memory usage")
        return gpu_id
    except Exception as e:
        print(f"Error getting GPU usage: {e}")
        return 0  # Fallback to GPU 0 if there's an error

class MaterialDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
class MaterialPropertyMLP(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(MaterialPropertyMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)
    
class ResNet50Model(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ResNet50Model, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Modify the first layer to accept our input dimension
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final layer for our regression task
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        # Reshape input to match ResNet50's expected input shape
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1, 1)  # Reshape to [batch_size, 1, features, 1]
        x = x.expand(-1, 1, -1, 224)  # Expand to match ResNet50's expected size
        return self.resnet(x)

def train_resnet50(X_train, X_test, y_train, y_test, target='band_gap', 
                  batch_size=32, num_epochs=50, learning_rate=0.001):
    """
    Train a ResNet50-based model for material property prediction.
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing targets
        target: Target property name
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    # Select the least utilized GPU
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

    gpu_id = get_least_utilized_gpu()
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Using GPU {gpu_id} for training")
    torch.cuda.empty_cache()
    # Create datasets
    train_dataset = MaterialDataset(X_train, y_train)
    test_dataset = MaterialDataset(X_test, y_test)
    
    # Create data loaders with multiple workers
    num_workers = 8#min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           num_workers=num_workers, pin_memory=True)
    
    # Initialize model
    model = ResNet50Model(input_dim=X_train.shape[1])
    #model = MaterialPropertyMLP(input_dim=X_train.shape[1])
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        # Create progress bar for batches
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for inputs, targets in batch_pbar:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            print(torch.cuda.memory_summary())
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
        
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'test_loss': f'{test_loss:.4f}'
        })
    
    # Save the model
    torch.save(model.state_dict(), f'models/resnet50_{target}.pth')
    
    return model, train_losses, test_losses

def predict_resnet50(model, X):
    """
    Make predictions using the trained ResNet50 model.
    
    Args:
        model: Trained ResNet50 model
        X: Input features
    """
    model.eval()
    gpu_id = get_least_utilized_gpu()
    device = torch.device(f'cuda:{gpu_id}')
    model = model.to(device)
    
    with torch.no_grad():
        X = torch.FloatTensor(X).to(device)
        predictions = model(X)
    
    return predictions.cpu().numpy() 