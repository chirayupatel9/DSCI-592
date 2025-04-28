import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from PIL import Image
import numpy as np

class MaterialPropertyPredictor(nn.Module):
    def __init__(self, 
                 num_metadata_features,
                 num_classes=None,
                 regression_target=False,
                 pretrained=True):
        super(MaterialPropertyPredictor, self).__init__()
        
        # Image processing branch (ResNet18)
        self.image_branch = models.resnet18(pretrained=pretrained)
        # Remove the final fully connected layer
        self.image_branch = nn.Sequential(*list(self.image_branch.children())[:-1])
        
        # Metadata processing branch
        self.metadata_branch = nn.Sequential(
            nn.Linear(num_metadata_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 256),  # 512 from ResNet18 + 256 from metadata
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Final prediction head
        if regression_target:
            self.prediction_head = nn.Linear(128, 1)
        else:
            self.prediction_head = nn.Linear(128, num_classes)
        
        self.regression_target = regression_target
        
    def forward(self, image, metadata):
        # Process image
        image_features = self.image_branch(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        # Process metadata
        metadata_features = self.metadata_branch(metadata)
        
        # Concatenate features
        combined_features = torch.cat([image_features, metadata_features], dim=1)
        
        # Fusion
        fused_features = self.fusion(combined_features)
        
        # Final prediction
        output = self.prediction_head(fused_features)
        
        if not self.regression_target:
            output = F.softmax(output, dim=1)
            
        return output

class MaterialDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 image_paths, 
                 metadata_list, 
                 targets, 
                 transform=None,
                 target_transform=None):
        self.image_paths = image_paths
        self.metadata_list = metadata_list
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Get metadata and ensure float32
        metadata = torch.tensor(self.metadata_list[idx], dtype=torch.float32)
        
        # Get target and ensure correct dtype
        target = self.targets[idx]
        if self.target_transform:
            target = self.target_transform(target)
            
        # Convert target to tensor with appropriate dtype
        if isinstance(target, np.ndarray):
            # Handle numpy arrays
            if target.dtype in (np.float32, np.float64):
                target = torch.tensor(target, dtype=torch.float32)
            elif target.dtype in (np.int32, np.int64):
                target = torch.tensor(target, dtype=torch.long)
            else:
                raise TypeError(f"Unsupported numpy array dtype: {target.dtype}")
        elif isinstance(target, (float, np.float32, np.float64)):
            target = torch.tensor(target, dtype=torch.float32)
        elif isinstance(target, (int, np.int32, np.int64)):
            target = torch.tensor(target, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported target type: {type(target)}")
            
        return image, metadata, target

def train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                device, 
                num_epochs=10,
                regression_target=False):
    print(f"Training model on {device}")
    best_val_loss = float('inf')
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (images, metadata, targets) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, metadata)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print GPU memory usage every 10 batches
            if device.type == 'cuda' and batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, metadata, targets in val_loader:
                # Move data to device
                images = images.to(device)
                metadata = metadata.to(device)
                targets = targets.to(device)
                
                outputs = model(images, metadata)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        if device.type == 'cuda':
            print(f'GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
    return model 