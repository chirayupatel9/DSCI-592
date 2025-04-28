"""
A simple API to predict material properties using the trained machine learning models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
import io
from dbconfig import MONGO_URI, MONGO_DB, MONGO_JSON_COLLECTION, MONGO_IMAGE_COLLECTION
from pymongo import MongoClient

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
json_collection = db[MONGO_JSON_COLLECTION]
image_collection = db[MONGO_IMAGE_COLLECTION]

# Load trained models
model_dir = "models"
models = {}

def load_models():
    """Load all trained models"""
    print("Loading trained models...")
    
    # Get all joblib files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    
    for model_file in model_files:
        target = model_file.split('_')[-1].replace('.joblib', '')
        model_path = os.path.join(model_dir, model_file)
        
        try:
            model = joblib.load(model_path)
            
            # Store model by target and algorithm
            if target not in models:
                models[target] = {}
            
            # Extract model type
            if 'random_forest' in model_file:
                models[target]['random_forest'] = model
            elif 'gradient_boosting' in model_file:
                models[target]['gradient_boosting'] = model
            elif 'elastic_net' in model_file:
                models[target]['elastic_net'] = model
            
            print(f"Loaded model: {model_file}")
            
        except Exception as e:
            print(f"Error loading model {model_file}: {str(e)}")
    
    print(f"Loaded {len(models)} models")

def get_best_model(target):
    """Get the best model for a target property"""
    # Look for a file indicating the best model
    best_model_path = os.path.join(model_dir, f"best_model_{target}.txt")
    
    model_type = 'random_forest'  # Default
    
    if os.path.exists(best_model_path):
        with open(best_model_path, 'r') as f:
            model_type = f.read().strip()
    
    return models.get(target, {}).get(model_type)

def prepare_element_features(elements):
    """Prepare the element presence features"""
    # List of possible elements
    element_set = set(["H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", 
                  "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", 
                  "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb", "Sr", "Y", "Zr", "Nb", 
                  "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", 
                  "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", 
                  "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", 
                  "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
                  "Pa", "U", "Np", "Pu"])
    
    element_features = {}
    for element in element_set:
        element_features[f"contains_{element}"] = element in elements
    
    return element_features

def prepare_crystal_features(crystal_system):
    """Prepare the crystal system features"""
    crystal_systems = ["Cubic", "Hexagonal", "Monoclinic", "Orthorhombic", 
                      "Tetragonal", "Triclinic", "Trigonal"]
    
    crystal_features = {}
    for system in crystal_systems:
        crystal_features[f"crystal_{system}"] = int(crystal_system == system)
    
    return crystal_features

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict material properties"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract material information
        material_id = data.get('material_id')
        formula = data.get('formula')
        
        # If material_id is provided, look up in the database
        if material_id:
            doc = json_collection.find_one({"material_id": material_id})
            if doc:
                # Prepare result with actual values where available
                result = {
                    "material_id": material_id,
                    "formula": doc.get("formula_pretty", ""),
                    "predictions": {},
                    "actual_values": {}
                }
                
                # Add actual values if available
                for target in models:
                    if target in doc:
                        result["actual_values"][target] = doc[target]
                
                # Prepare features for prediction
                elements = doc.get("elements", [])
                crystal_system = doc.get("symmetry", {}).get("crystal_system", "")
                
                # Create feature dictionary with exact feature names
                features = {
                    "nelements": doc.get("nelements", 0),
                    "nsites": doc.get("nsites", 0),
                    "density": doc.get("density", 0),
                    "volume": doc.get("volume", 0),
                    "formation_energy_per_atom": doc.get("formation_energy_per_atom", 0),
                    "energy_per_atom": doc.get("energy_per_atom", 0)
                }
                
                # Add element features with exact feature names
                element_features = prepare_element_features(elements)
                for element in element_features:
                    features[element] = element_features[element]
                
                # Add crystal system features with exact feature names
                crystal_features = prepare_crystal_features(crystal_system)
                for crystal in crystal_features:
                    features[crystal] = crystal_features[crystal]
                
                # Make predictions for each target
                for target in models:
                    # Skip if target is in the features (to avoid using the target as a feature)
                    if target in features:
                        feature_copy = features.copy()
                        feature_copy.pop(target)
                        features_df = pd.DataFrame([feature_copy])
                    else:
                        features_df = pd.DataFrame([features])
                    
                    # Get the model
                    model = get_best_model(target)
                    if model:
                        try:
                            # Ensure feature names match exactly
                            feature_names = model.feature_names_in_
                            features_df = features_df[feature_names]
                            prediction = model.predict(features_df)[0]
                            result["predictions"][target] = float(prediction)
                        except Exception as e:
                            print(f"Error predicting {target}: {str(e)}")
                            result["predictions"][target] = None
                
                return jsonify(result)
            
            else:
                return jsonify({"error": f"Material with ID {material_id} not found"}), 404
        
        # If formula is provided, create predictions from scratch
        elif formula:
            # Extract elements from formula
            import re
            elements = re.findall(r'([A-Z][a-z]*)(?:\d*\.?\d*)?', formula)
            
            # Get crystal_system if provided or default to empty
            crystal_system = data.get('crystal_system', '')
            
            # Create feature dictionary
            features = {
                "nelements": len(elements),
                "nsites": data.get('nsites', 0),
                "density": data.get('density', 0),
                "volume": data.get('volume', 0),
                "formation_energy_per_atom": data.get('formation_energy_per_atom', 0),
                "energy_per_atom": data.get('energy_per_atom', 0)
            }
            
            # Add element features
            features.update(prepare_element_features(elements))
            
            # Add crystal system features
            features.update(prepare_crystal_features(crystal_system))
            
            # Create result dictionary
            result = {
                "formula": formula,
                "predictions": {}
            }
            
            # Make predictions for each target
            for target in models:
                # Skip if target is in the features (to avoid using the target as a feature)
                if target in features:
                    feature_copy = features.copy()
                    feature_copy.pop(target)
                    features_df = pd.DataFrame([feature_copy])
                else:
                    features_df = pd.DataFrame([features])
                
                # Get the model
                model = get_best_model(target)
                if model:
                    prediction = model.predict(features_df)[0]
                    result["predictions"][target] = float(prediction)
            
            return jsonify(result)
        
        else:
            return jsonify({"error": "Either material_id or formula must be provided"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/materials/<material_id>', methods=['GET'])
def get_material(material_id):
    """API endpoint to get material information"""
    try:
        doc = json_collection.find_one({"material_id": material_id})
        if doc:
            # Convert ObjectId to string for JSON serialization
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            
            # Add image information if available
            if 'image_id' in doc:
                image_doc = image_collection.find_one({"image_id": doc['image_id']}, 
                                                    {"_id": 0, "binary_data": 0})
                if image_doc:
                    doc['image_info'] = image_doc
                    doc['image_url'] = f"/materials/{material_id}/image"
            
            return jsonify(doc)
        else:
            return jsonify({"error": f"Material with ID {material_id} not found"}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/materials/<material_id>/image', methods=['GET'])
def get_material_image(material_id):
    """API endpoint to get material image"""
    try:
        # First get the material document to find the image_id
        material = json_collection.find_one({"material_id": material_id})
        if not material or 'image_id' not in material:
            return jsonify({"error": "Material not found or has no associated image"}), 404
        
        # Get the image document
        image_doc = image_collection.find_one({"image_id": material['image_id']})
        if not image_doc or 'binary_data' not in image_doc:
            return jsonify({"error": "Image not found"}), 404
        
        # Create a file-like object from the binary data
        image_data = io.BytesIO(image_doc['binary_data'])
        
        # Send the file with the correct mimetype
        mimetype = f"image/{image_doc['format'].lower()}"
        return send_file(image_data, mimetype=mimetype)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['GET'])
def search_materials():
    """API endpoint to search for materials"""
    try:
        # Get query parameters
        query = {}
        
        # Formula search
        formula = request.args.get('formula')
        if formula:
            query['formula_pretty'] = {'$regex': formula, '$options': 'i'}
        
        # Elements search
        elements = request.args.get('elements')
        if elements:
            element_list = elements.split(',')
            query['elements'] = {'$all': element_list}
        
        # Band gap range
        min_band_gap = request.args.get('min_band_gap')
        max_band_gap = request.args.get('max_band_gap')
        if min_band_gap or max_band_gap:
            query['band_gap'] = {}
            if min_band_gap:
                query['band_gap']['$gte'] = float(min_band_gap)
            if max_band_gap:
                query['band_gap']['$lte'] = float(max_band_gap)
        
        # Crystal system
        crystal_system = request.args.get('crystal_system')
        if crystal_system:
            query['symmetry.crystal_system'] = crystal_system
        
        # Has image filter
        has_image = request.args.get('has_image')
        if has_image and has_image.lower() == 'true':
            query['image_id'] = {'$exists': True}
        
        # Limit results
        limit = int(request.args.get('limit', 10))
        
        results = []
        cursor = json_collection.find(query).limit(limit)
        
        for doc in cursor:
            # Convert ObjectId to string for JSON serialization
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            
            # Add image information if available
            if 'image_id' in doc:
                image_doc = image_collection.find_one({"image_id": doc['image_id']}, 
                                                    {"_id": 0, "binary_data": 0})
                if image_doc:
                    doc['image_info'] = image_doc
                    doc['image_url'] = f"/materials/{doc['material_id']}/image"
            
            results.append(doc)
        
        return jsonify({"count": len(results), "results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        "api": "Materials Property Prediction API",
        "endpoints": {
            "/predict": "POST - Predict material properties",
            "/materials/<material_id>": "GET - Get material information",
            "/search": "GET - Search for materials",
            "/": "GET - API information"
        },
        "targets": list(models.keys())
    })

if __name__ == "__main__":
    # Load models before starting the API
    load_models()
    
    # Start API server
    app.run(debug=True, host='0.0.0.0', port=5001) 