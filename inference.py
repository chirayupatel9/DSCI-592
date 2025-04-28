import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
from model import MaterialPropertyPredictor
from pymongo import MongoClient
import os

def load_model(model_path, metadata_path):
    """Load trained model and its metadata"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model = MaterialPropertyPredictor(
        num_metadata_features=metadata['num_metadata_features'],
        num_classes=metadata['num_classes'],
        regression_target=metadata['regression_target']
    )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, metadata

def preprocess_metadata(doc):
    """Preprocess metadata for inference (same as in training)"""
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

def predict_material_property(material_id, target_property='band_gap'):
    """Predict material property for a given material ID"""
    # Load model and metadata
    model_path = f'final_model_{target_property}.pth'
    metadata_path = f'model_metadata_{target_property}.json'
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Model files not found. Please train the model first.")
    
    model, metadata = load_model(model_path, metadata_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load material data from MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['materials_db']
    collection = db['materials']
    
    doc = collection.find_one({"material_id": material_id})
    if not doc:
        raise ValueError(f"Material with ID {material_id} not found")
    
    # Get image path
    if 'image_id' not in doc:
        raise ValueError("Material has no associated image")
    
    image_path = f"data/output/png/{doc['image_id']}.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    # Preprocess metadata
    metadata_features = preprocess_metadata(doc)
    metadata_features = torch.tensor(metadata_features, dtype=torch.float32).unsqueeze(0)
    
    # Move to device
    image = image.to(device)
    metadata_features = metadata_features.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image, metadata_features)
    
    if metadata['regression_target']:
        prediction = output.item()
    else:
        prediction = torch.argmax(output, dim=1).item()
    
    return {
        'material_id': material_id,
        'target_property': target_property,
        'prediction': prediction,
        'actual_value': doc.get(target_property, None)
    }

if __name__ == '__main__':
    # Example usage
    try:
        result = predict_material_property('mp-10004', 'band_gap')
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}") 