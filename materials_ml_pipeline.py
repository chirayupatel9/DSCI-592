# %% [markdown]
# # Materials Science Machine Learning Pipeline
# 
# This notebook contains the complete implementation of the materials science machine learning pipeline, including data processing, model training, API implementation, and visualization.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pymongo
from pymongo import MongoClient
import json
import os
import pickle
from flask import Flask, request, jsonify
import requests
from PIL import Image
import io
import base64
from typing import Dict, List, Union
import logging
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn')
sns.set_palette('husl')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %% [markdown]
# ## 2. Data Collection and Processing

# %%
class DataProcessor:
    def __init__(self, mongo_uri: str = 'mongodb://localhost:27017/'):
        self.client = MongoClient(mongo_uri)
        self.db = self.client['materials_db']
        self.collection = self.db['materials']
        
    def load_data(self) -> pd.DataFrame:
        """Load data from MongoDB"""
        try:
            data = pd.DataFrame(list(self.collection.find()))
            logger.info(f"Loaded {len(data)} records from MongoDB")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def process_elements(self, elements_str):
        """Convert elements string into numeric features"""
        from collections import Counter
        elements = elements_str.split(',')
        element_counts = Counter(elements)
        # Convert counts to binary (0 or 1)
        return {f'element_{element}': 1 for element in element_counts}
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess the materials data"""
        try:
            # Handle missing values
            df = df.dropna(subset=['band_gap', 'formation_energy'])
            
            # Process elements into features
            element_features = pd.DataFrame([self.process_elements(elem) for elem in df['formula']])
            element_features = element_features.fillna(0)  # Fill NaN with 0 for elements not present
            
            # Convert boolean columns to 0/1
            bool_columns = [col for col in df.columns if df[col].dtype == bool]
            df[bool_columns] = df[bool_columns].astype(int)
            
            # Feature engineering
            df['avg_atomic_radius'] = df['atomic_radii'].apply(lambda x: np.mean(x))
            df['total_electrons'] = df['element_counts'].apply(lambda x: sum(x.values()))
            
            # Select features and target
            features = ['avg_atomic_radius', 'total_electrons', 'crystal_system', 'space_group']
            target = 'band_gap'
            
            # Convert categorical variables
            df = pd.get_dummies(df, columns=['crystal_system', 'space_group'])
            
            # Combine original features with element features
            df = pd.concat([df, element_features], axis=1)
            
            return df, df[target]
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to file"""
        try:
            df.to_csv(filename, index=False)
            logger.info(f"Saved processed data to {filename}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

# %% [markdown]
# ## 3. Model Implementation

# %%
class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_params = {}
        
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
        """Train Random Forest model with hyperparameter tuning"""
        try:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
            grid_search.fit(X, y)
            
            self.best_params['random_forest'] = grid_search.best_params_
            self.models['random_forest'] = grid_search.best_estimator_
            
            logger.info(f"Random Forest best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        except Exception as e:
            logger.error(f"Error training Random Forest: {str(e)}")
            raise
    
    def train_gradient_boosting(self, X: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
        """Train Gradient Boosting model with hyperparameter tuning"""
        try:
            param_grid = {
                'n_estimators': [200, 500, 1000],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
            
            gb = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='r2')
            grid_search.fit(X, y)
            
            self.best_params['gradient_boosting'] = grid_search.best_params_
            self.models['gradient_boosting'] = grid_search.best_estimator_
            
            logger.info(f"Gradient Boosting best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        except Exception as e:
            logger.error(f"Error training Gradient Boosting: {str(e)}")
            raise
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model performance"""
        try:
            y_pred = model.predict(X_test)
            
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': np.mean(np.abs(y_test - y_pred))
            }
            
            logger.info(f"Model metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model, filename: str):
        """Save trained model to file"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model to {filename}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filename: str):
        """Load trained model from file"""
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded model from {filename}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

# %% [markdown]
# ## 4. API Implementation

# %%
class PredictionAPI:
    def __init__(self, model_path: str):
        self.app = Flask(__name__)
        self.model = self.load_model(model_path)
        self.setup_routes()
        
    def load_model(self, model_path: str):
        """Load the trained model"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def setup_routes(self):
        """Setup API routes"""
        @self.app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.get_json()
                features = self.preprocess_input(data)
                prediction = self.model.predict(features)
                
                response = {
                    'prediction': float(prediction[0]),
                    'timestamp': datetime.now().isoformat(),
                    'model_version': '1.0.0'
                }
                
                return jsonify(response)
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy'})
    
    def preprocess_input(self, data: dict) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        try:
            # Convert input data to DataFrame
            df = pd.DataFrame([data])
            
            # Apply same preprocessing as training
            df['avg_atomic_radius'] = df['atomic_radii'].apply(lambda x: np.mean(x))
            df['total_electrons'] = df['element_counts'].apply(lambda x: sum(x.values()))
            
            # Convert categorical variables
            df = pd.get_dummies(df, columns=['crystal_system', 'space_group'])
            
            return df
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise
    
    def run(self, host: str = '0.0.0.0', port: int = 5000):
        """Run the API server"""
        self.app.run(host=host, port=port, debug=True)

# %% [markdown]
# ## 5. Visualization

# %%
class Visualizer:
    def __init__(self, output_dir: str = 'plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_feature_importance(self, model, feature_names: List[str], title: str):
        """Plot feature importance"""
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{title.lower().replace(" ", "_")}.png'))
        plt.close()
    
    def plot_actual_vs_predicted(self, y_test: pd.Series, y_pred: np.ndarray, title: str):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{title.lower().replace(" ", "_")}.png'))
        plt.close()
    
    def plot_crystal_system_distribution(self, df: pd.DataFrame):
        """Plot distribution of crystal systems"""
        plt.figure(figsize=(10, 6))
        df['crystal_system'].value_counts().plot(kind='bar')
        plt.title('Crystal System Distribution')
        plt.xlabel('Crystal System')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'crystal_system_distribution.png'))
        plt.close()
    
    def plot_band_gap_distribution(self, df: pd.DataFrame):
        """Plot distribution of band gaps"""
        plt.figure(figsize=(10, 6))
        sns.histplot(df['band_gap'], bins=30)
        plt.title('Band Gap Distribution')
        plt.xlabel('Band Gap (eV)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'band_gap_distribution.png'))
        plt.close()
    
    def plot_element_distribution(self, df: pd.DataFrame):
        """Plot distribution of elements"""
        element_counts = {}
        for counts in df['element_counts']:
            for element, count in counts.items():
                if element in element_counts:
                    element_counts[element] += count
                else:
                    element_counts[element] = count
        
        plt.figure(figsize=(12, 6))
        pd.Series(element_counts).sort_values(ascending=False).head(20).plot(kind='bar')
        plt.title('Top 20 Elements Distribution')
        plt.xlabel('Element')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'element_distribution.png'))
        plt.close()
    
    def plot_property_correlation(self, df: pd.DataFrame):
        """Plot correlation between properties"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[['band_gap', 'formation_energy', 'avg_atomic_radius', 'total_electrons']].corr(),
                    annot=True, cmap='coolwarm', center=0)
        plt.title('Property Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'property_correlation.png'))
        plt.close()
    
    def plot_error_distribution(self, y_test: pd.Series, y_pred: np.ndarray, title: str):
        """Plot error distribution"""
        errors = y_test - y_pred
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=30)
        plt.title(f'{title} Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{title.lower().replace(" ", "_")}_error_distribution.png'))
        plt.close()

# %% [markdown]
# ## 6. Main Pipeline

# %%
def main():
    # Initialize components
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    visualizer = Visualizer()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = data_processor.load_data()
    X, y = data_processor.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    logger.info("Training Random Forest...")
    rf_model = model_trainer.train_random_forest(X_train, y_train)
    
    logger.info("Training Gradient Boosting...")
    gb_model = model_trainer.train_gradient_boosting(X_train, y_train)
    
    # Evaluate models
    logger.info("Evaluating models...")
    rf_metrics = model_trainer.evaluate_model(rf_model, X_test, y_test)
    gb_metrics = model_trainer.evaluate_model(gb_model, X_test, y_test)
    
    # Generate predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_gb = gb_model.predict(X_test)
    
    # Create visualizations
    logger.info("Generating visualizations...")
    visualizer.plot_feature_importance(rf_model, X.columns, 'Random Forest Feature Importance')
    visualizer.plot_feature_importance(gb_model, X.columns, 'Gradient Boosting Feature Importance')
    
    visualizer.plot_actual_vs_predicted(y_test, y_pred_rf, 'Random Forest Predictions')
    visualizer.plot_actual_vs_predicted(y_test, y_pred_gb, 'Gradient Boosting Predictions')
    
    visualizer.plot_crystal_system_distribution(df)
    visualizer.plot_band_gap_distribution(df)
    visualizer.plot_element_distribution(df)
    visualizer.plot_property_correlation(df)
    
    visualizer.plot_error_distribution(y_test, y_pred_rf, 'Random Forest')
    visualizer.plot_error_distribution(y_test, y_pred_gb, 'Gradient Boosting')
    
    # Save models
    logger.info("Saving models...")
    model_trainer.save_model(rf_model, 'models/random_forest.pkl')
    model_trainer.save_model(gb_model, 'models/gradient_boosting.pkl')
    
    # Initialize and run API
    logger.info("Starting API server...")
    api = PredictionAPI('models/random_forest.pkl')
    api.run()

if __name__ == "__main__":
    main() 