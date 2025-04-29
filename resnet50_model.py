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

class MaterialDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

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
    # Create datasets
    train_dataset = MaterialDataset(X_train, y_train)
    test_dataset = MaterialDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = ResNet50Model(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
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
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
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
    device = next(model.parameters()).device
    
    with torch.no_grad():
        X = torch.FloatTensor(X).to(device)
        predictions = model(X)
    
    return predictions.cpu().numpy() 