"""
Explore and analyze the materials science data in MongoDB.
This script performs basic data exploration and prepares data for machine learning.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from dbconfig import MONGO_URI, MONGO_DB, MONGO_JSON_COLLECTION, MONGO_IMAGE_COLLECTION

def connect_to_mongodb():
    """Connect to MongoDB and return database instance"""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    return client, db

def explore_json_data(db):
    """Explore the JSON material data in MongoDB"""
    collection = db[MONGO_JSON_COLLECTION]
    
    # Get collection stats
    doc_count = collection.count_documents({})
    print(f"Total documents in {MONGO_JSON_COLLECTION}: {doc_count}")
    
    # Explore available fields
    if doc_count > 0:
        sample_doc = collection.find_one({})
        print("\nSample document fields:")
        for field in sorted(sample_doc.keys()):
            print(f"- {field}")
    
    # Check for missing values in important fields
    print("\nMissing values in important fields:")
    important_fields = ["material_id", "formula_pretty", "elements", "band_gap", "density"]
    for field in important_fields:
        missing_count = collection.count_documents({field: {"$exists": False}})
        print(f"- {field}: {missing_count} documents ({missing_count/doc_count*100:.2f}%)")
    
    # Basic statistics on numeric properties
    numeric_fields = ["band_gap", "density", "energy_per_atom", "formation_energy_per_atom"]
    print("\nBasic statistics on important numeric fields:")
    
    for field in numeric_fields:
        # Skip non-existent fields
        if collection.count_documents({field: {"$exists": True}}) == 0:
            print(f"- {field}: No data available")
            continue
            
        # Get statistics
        pipeline = [
            {"$match": {field: {"$exists": True, "$ne": None}}},
            {"$group": {
                "_id": None,
                "avg": {"$avg": f"${field}"},
                "min": {"$min": f"${field}"},
                "max": {"$max": f"${field}"},
                "count": {"$sum": 1}
            }}
        ]
        
        result = list(collection.aggregate(pipeline))
        if result:
            stats = result[0]
            print(f"- {field}: count={stats['count']}, min={stats['min']:.4f}, avg={stats['avg']:.4f}, max={stats['max']:.4f}")
        else:
            print(f"- {field}: No data available")
    
    # Count materials by elements
    element_counts = {}
    for doc in collection.find({}, {"elements": 1}):
        if "elements" in doc:
            for elem in doc["elements"]:
                element_counts[elem] = element_counts.get(elem, 0) + 1
    
    print("\nMost common elements in the dataset:")
    for elem, count in sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"- {elem}: {count} materials ({count/doc_count*100:.2f}%)")

def explore_image_data(db):
    """Explore the image data in MongoDB"""
    collection = db[MONGO_IMAGE_COLLECTION]
    
    # Get collection stats
    img_count = collection.count_documents({})
    print(f"\nTotal images in {MONGO_IMAGE_COLLECTION}: {img_count}")
    
    # Image format distribution
    format_counts = {}
    for doc in collection.find({}, {"format": 1}):
        if "format" in doc:
            fmt = doc["format"]
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    print("\nImage format distribution:")
    for fmt, count in sorted(format_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {fmt}: {count} images ({count/img_count*100:.2f}%)")
    
    # Image dimensions
    widths = []
    heights = []
    
    for doc in collection.find({}, {"width": 1, "height": 1}):
        if "width" in doc and "height" in doc:
            widths.append(doc["width"])
            heights.append(doc["height"])
    
    if widths and heights:
        print("\nImage dimensions statistics:")
        print(f"- Width: min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.2f}")
        print(f"- Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.2f}")

def create_data_for_ml(db):
    """Prepare data for machine learning tasks"""
    json_collection = db[MONGO_JSON_COLLECTION]
    
    # Create a dataframe with important features for ML
    features = []
    
    # Define all possible elements
    element_set = set(["H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", 
                      "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", 
                      "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb", "Sr", "Y", "Zr", "Nb", 
                      "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", 
                      "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", 
                      "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", 
                      "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
                      "Pa", "U", "Np", "Pu"])
    
    for doc in json_collection.find():
        try:
            # Collect basic material information
            material_info = {
                "material_id": doc.get("material_id", ""),
                "formula": doc.get("formula_pretty", ""),
                "nelements": doc.get("nelements", 0),
                "nsites": doc.get("nsites", 0),
                "density": doc.get("density", 0),
                "volume": doc.get("volume", 0),
                "band_gap": doc.get("band_gap", 0),
                "is_metal": int(doc.get("is_metal", False)),  # Convert boolean to int
                "formation_energy_per_atom": doc.get("formation_energy_per_atom", 0),
                "energy_per_atom": doc.get("energy_per_atom", 0)
            }
            
            # Add crystal system if available
            if "symmetry" in doc and "crystal_system" in doc["symmetry"]:
                material_info["crystal_system"] = doc["symmetry"]["crystal_system"]
            else:
                material_info["crystal_system"] = ""
            
            # Extract elements and create numerical features
            elements = doc.get("elements", [])
            
            # Add element features (binary 0/1)
            for element in element_set:
                material_info[f"element_{element}"] = int(element in elements)
            
            # Add dielectric constant if available
            if "dielectric" in doc and "n" in doc["dielectric"]:
                material_info["dielectric_constant"] = doc["dielectric"]["n"]
            else:
                material_info["dielectric_constant"] = None
            
            features.append(material_info)
            
        except Exception as e:
            print(f"Error processing document {doc.get('material_id', 'unknown')}: {str(e)}")
    
    # Convert to dataframe
    df = pd.DataFrame(features)
    
    # Convert any remaining boolean columns to integers
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)
    
    # Save to CSV for ML tasks
    df.to_csv("materials_for_ml.csv", index=False)
    print(f"\nSaved {len(df)} material records to 'materials_for_ml.csv' for machine learning")
    
    # Print summary statistics
    print("\nDataset summary statistics:")
    print(df.describe())
    
    # Display head
    print("\nDataset preview (first 5 rows):")
    print(df.head())
    
    return df

def plot_basic_statistics(df):
    """Create some basic plots for data visualization"""
    if df.empty:
        print("DataFrame is empty, skipping plots.")
        return
    
    os.makedirs("plots", exist_ok=True)
    
    # Set plot style
    plt.style.use('ggplot')
    
    # Plot 1: Band gap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['band_gap'], bins=30, alpha=0.7, color='blue')
    plt.title('Distribution of Band Gap Values')
    plt.xlabel('Band Gap (eV)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plots/band_gap_distribution.png')
    
    # Plot 2: Formation energy vs band gap
    plt.figure(figsize=(10, 6))
    plt.scatter(df['formation_energy_per_atom'], df['band_gap'], alpha=0.5)
    plt.title('Formation Energy vs Band Gap')
    plt.xlabel('Formation Energy per Atom (eV)')
    plt.ylabel('Band Gap (eV)')
    plt.tight_layout()
    plt.savefig('plots/formation_energy_vs_band_gap.png')
    
    # Plot 3: Number of elements distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['nelements'], bins=range(1, df['nelements'].max() + 2), alpha=0.7, color='green')
    plt.title('Distribution of Number of Elements')
    plt.xlabel('Number of Elements')
    plt.ylabel('Count')
    plt.xticks(range(1, df['nelements'].max() + 1))
    plt.tight_layout()
    plt.savefig('plots/nelements_distribution.png')
    
    # Plot 4: Crystal system distribution
    if 'crystal_system' in df.columns:
        crystal_counts = df['crystal_system'].value_counts()
        plt.figure(figsize=(12, 6))
        crystal_counts.plot(kind='bar', color='purple')
        plt.title('Distribution of Crystal Systems')
        plt.xlabel('Crystal System')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('plots/crystal_system_distribution.png')
    
    print("\nPlots saved to 'plots' directory")

def main():
    """Main function to explore the MongoDB data"""
    client, db = connect_to_mongodb()
    
    try:
        print("Starting data exploration...")
        explore_json_data(db)
        explore_image_data(db)
        
        print("\nPreparing data for machine learning...")
        df = create_data_for_ml(db)
        
        print("\nCreating visualizations...")
        plot_basic_statistics(df)
        
        print("Data exploration completed successfully")
    
    except Exception as e:
        print(f"Error during data exploration: {str(e)}")
    
    finally:
        client.close()
        print("MongoDB connection closed")

if __name__ == "__main__":
    main() 