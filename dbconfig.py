"""
Database configuration for MongoDB connection
"""

from pymongo import MongoClient

# MongoDB Configuration
MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_URI = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/"
MONGO_DB = "materials_science_db_01"
MONGO_JSON_COLLECTION = "material_data"
MONGO_IMAGE_COLLECTION = "material_images"
DATA_FOLDER = "data"
