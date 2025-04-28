"""
Script to load the materials science data into MongoDB database.
This script processes both JSON data files and image files.
"""

import os
import json
import base64
from PIL import Image
import io
from tqdm import tqdm
import pymongo
from dbconfig import MONGO_URI, MONGO_DB, MONGO_JSON_COLLECTION, MONGO_IMAGE_COLLECTION, DATA_FOLDER

def connect_to_mongodb():
    """Connect to MongoDB and return database instance"""
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    return client, db

def load_json_data(db):
    """Load JSON data files into MongoDB"""
    collection = db[MONGO_JSON_COLLECTION]
    
    # Create indexes for faster queries
    collection.create_index("material_id")
    collection.create_index("ID_name")
    collection.create_index("elements")
    collection.create_index("formula_pretty")
    collection.create_index("image_id")  # Add index for image reference
    
    json_dir = os.path.join(DATA_FOLDER, "json")
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    print(f"Loading {len(json_files)} JSON files to MongoDB...")
    
    for json_file in tqdm(json_files):
        try:
            file_path = os.path.join(json_dir, json_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add file_id field to track original file
            data['file_id'] = os.path.splitext(json_file)[0]
            
            # Add image_id field (assuming image files have the same base name)
            data['image_id'] = data['file_id']
            
            # Check if document already exists
            existing = collection.find_one({"file_id": data['file_id']})
            if existing:
                collection.replace_one({"file_id": data['file_id']}, data)
            else:
                collection.insert_one(data)
                
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    print(f"Loaded {collection.count_documents({})} documents to {MONGO_JSON_COLLECTION} collection")

def load_image_data(db):
    """Load image files into MongoDB"""
    collection = db[MONGO_IMAGE_COLLECTION]
    
    # Create indexes
    collection.create_index("image_id", unique=True)
    
    img_dir = os.path.join(DATA_FOLDER, "png")
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Loading {len(img_files)} image files to MongoDB...")
    
    for img_file in tqdm(img_files):
        try:
            file_path = os.path.join(img_dir, img_file)
            image_id = os.path.splitext(img_file)[0]
            
            # Check if image already exists
            existing = collection.find_one({"image_id": image_id})
            if existing:
                continue
            
            # Read image as binary
            with open(file_path, 'rb') as f:
                img_binary = f.read()
            
            # Get image metadata
            img = Image.open(file_path)
            width, height = img.size
            format_type = img.format
            mode = img.mode
            
            # Create document
            image_doc = {
                "image_id": image_id,
                "filename": img_file,
                "width": width,
                "height": height,
                "format": format_type,
                "mode": mode,
                "binary_data": img_binary
            }
            
            collection.insert_one(image_doc)
                
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    print(f"Loaded {collection.count_documents({})} images to {MONGO_IMAGE_COLLECTION} collection")

def verify_image_links(db):
    """Verify and report the links between JSON data and images"""
    json_collection = db[MONGO_JSON_COLLECTION]
    image_collection = db[MONGO_IMAGE_COLLECTION]
    
    total_materials = json_collection.count_documents({})
    materials_with_images = json_collection.count_documents({"image_id": {"$exists": True}})
    linked_images = image_collection.count_documents({
        "image_id": {"$in": json_collection.distinct("image_id")}
    })
    
    print("\nVerifying JSON-Image links:")
    print(f"Total materials: {total_materials}")
    print(f"Materials with image references: {materials_with_images}")
    print(f"Images linked to materials: {linked_images}")
    
    # Check for materials with missing images
    materials_missing_images = json_collection.count_documents({
        "image_id": {"$exists": True},
        "image_id": {"$nin": image_collection.distinct("image_id")}
    })
    if materials_missing_images > 0:
        print(f"Warning: {materials_missing_images} materials have missing image references")

def main():
    """Main function to load all data into MongoDB"""
    client, db = connect_to_mongodb()
    
    try:
        print("Starting data loading process...")
        load_json_data(db)
        load_image_data(db)
        verify_image_links(db)
        print("Data loading completed successfully")
    
    except Exception as e:
        print(f"Error during data loading: {str(e)}")
    
    finally:
        client.close()
        print("MongoDB connection closed")

if __name__ == "__main__":
    main() 