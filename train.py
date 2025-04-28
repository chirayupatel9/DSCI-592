import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from pymongo import MongoClient
import numpy as np
from PIL import Image
from model import MaterialPropertyPredictor, MaterialDataset, train_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
# from dbconfig import MONGO_URI, MONGO_DB, MONGO_JSON_COLLECTION, MONGO_IMAGE_COLLECTION

MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_URI = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/"
MONGO_DB = "materials_science_db_02"
MONGO_JSON_COLLECTION = "material_data"
MONGO_IMAGE_COLLECTION = "material_images"
DATA_FOLDER = "data/output"

def check_mongodb_connection():
    """Check MongoDB connection and database contents"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[MONGO_JSON_COLLECTION]
        
        # Check if database exists
        if MONGO_DB not in client.list_database_names():
            raise ValueError(f"Database '{MONGO_DB}' does not exist")
            
        # Check if collection exists
        if MONGO_JSON_COLLECTION not in db.list_collection_names():
            raise ValueError(f"Collection '{MONGO_JSON_COLLECTION}' does not exist")
            
        # Count documents
        doc_count = collection.count_documents({})
        print(f"Found {doc_count} documents in the collection")
        
        # Check for documents with images
        docs_with_images = collection.count_documents({'image_id': {'$exists': True}})
        print(f"Found {docs_with_images} documents with images")
        
        return client, db, collection
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        raise

def check_images_directory():
    """Check if images directory exists and contains files"""
    images_dir = "data/output/png"
    if not os.path.exists(images_dir):
        print(f"Creating images directory: {images_dir}")
        os.makedirs(images_dir)
        return False
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    print(f"Found {len(image_files)} image files in {images_dir}")
    return len(image_files) > 0

def preprocess_metadata(doc):
    """Extract and preprocess metadata features"""
    features = []
    
    # Numerical features
    numerical_features = [
        'density',
        'volume',
        'formation_energy_per_atom',
        'energy_above_hull',
        'band_gap'
    ]
    
    for feature in numerical_features:
        features.append(doc.get(feature, 0.0))
    
    # Crystal system (one-hot encoding)
    crystal_systems = ["Cubic", "Hexagonal", "Monoclinic", "Orthorhombic", 
                      "Tetragonal", "Triclinic", "Trigonal"]
    crystal_system = doc.get('symmetry', {}).get('crystal_system', '')
    for system in crystal_systems:
        features.append(1.0 if crystal_system == system else 0.0)
    
    # Elements (one-hot encoding)
    elements = ["H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", 
               "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", 
               "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb", "Sr", "Y", "Zr", "Nb", 
               "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", 
               "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", 
               "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", 
               "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
               "Pa", "U", "Np", "Pu"]
    
    material_elements = doc.get('elements', [])
    for element in elements:
        features.append(1.0 if element in material_elements else 0.0)
    
    return np.array(features)

def main():
    # Configuration
    target_property = 'band_gap'  # or 'formation_energy_per_atom', 'is_stable', 'magnetic_ordering'
    regression_target = target_property in ['band_gap', 'formation_energy_per_atom']
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    
    # Check MongoDB connection and data
    print("Checking MongoDB connection...")
    client, db, collection = check_mongodb_connection()
    
    # Check images directory
    print("\nChecking images directory...")
    if not check_images_directory():
        print("Warning: No images found in images directory")
        return
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load data from MongoDB
    print("\nLoading data from MongoDB...")
    image_paths = []
    metadata_list = []
    targets = []
    
    for doc in collection.find():
        if 'image_id' in doc and target_property in doc:
            image_path = os.path.join("data", "output", "png", f"{doc['image_id']}.png")
            if os.path.exists(image_path):
                image_paths.append(image_path)
                metadata_list.append(preprocess_metadata(doc))
                targets.append(doc[target_property])
    
    print(f"Found {len(image_paths)} valid samples with both images and target property")
    
    if len(image_paths) == 0:
        print("Error: No valid samples found. Please check:")
        print("1. MongoDB connection and data")
        print("2. Images directory and files")
        print("3. Target property field in documents")
        return
    
    # Convert targets to appropriate format
    if not regression_target:
        label_encoder = LabelEncoder()
        targets = label_encoder.fit_transform(targets)
        targets = targets.astype(np.int64)  # For classification
    else:
        targets = np.array(targets, dtype=np.float32)  # For regression
        targets = targets.reshape(-1, 1)  # Reshape for MSE loss compatibility
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = MaterialDataset(
        image_paths=image_paths,
        metadata_list=metadata_list,
        targets=targets,
        transform=transform
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    num_metadata_features = len(metadata_list[0])
    num_classes = None if regression_target else len(np.unique(targets))
    
    print("\nInitializing model...")
    model = MaterialPropertyPredictor(
        num_metadata_features=num_metadata_features,
        num_classes=num_classes,
        regression_target=regression_target
    )
    
    # Loss function and optimizer
    criterion = torch.nn.MSELoss() if regression_target else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print("\nStarting training...")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        regression_target=regression_target
    )
    
    # Save final model
    print("\nSaving model...")
    torch.save(model.state_dict(), f'final_model_{target_property}.pth')
    
    # Save metadata for inference
    metadata = {
        'target_property': target_property,
        'regression_target': regression_target,
        'num_classes': num_classes,
        'num_metadata_features': num_metadata_features
    }
    with open(f'model_metadata_{target_property}.json', 'w') as f:
        json.dump(metadata, f)
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main() 