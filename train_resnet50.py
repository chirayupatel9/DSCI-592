"""
Script to train the ResNet50 model for material property prediction.
This script uses the existing data preprocessing pipeline and trains the ResNet50 model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from resnet50_model import train_resnet50, predict_resnet50
import joblib
from collections import Counter

def convert_elements_to_features(elements_str):
    """Convert elements string into numeric features"""
    # Split elements and count occurrences
    elements = elements_str.split(',')
    element_counts = Counter(elements)
    
    # Create a dictionary of element counts
    features = {}
    for element in element_counts:
        features[f'element_{element}'] = element_counts[element]
    
    return features

def load_and_preprocess_data(target='band_gap'):
    """Load and preprocess data for ResNet50 model"""
    # Load data
    df = pd.read_csv("materials_for_ml.csv")
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target])
    
    # Convert elements to numeric features
    element_features = pd.DataFrame([convert_elements_to_features(elem) for elem in df['formula']])
    element_features = element_features.fillna(0)  # Fill NaN with 0 for elements not present
    
    # Select features
    categorical_cols = ["crystal_system"]
    element_cols = [col for col in df.columns if col.startswith('contains_')]
    numerical_cols = [col for col in df.columns if col not in categorical_cols + element_cols + [target, 'material_id', 'formula']]
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # Prepare features and target
    X = df.drop(columns=[target, 'material_id', 'formula'])
    
    # Combine original features with element features
    X = pd.concat([X, element_features], axis=1)
    
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, f'models/resnet50_{target}_scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values

def plot_training_history(train_losses, test_losses, target):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title(f'ResNet50 Training History - {target}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/resnet50_{target}_training_history.png')
    plt.close()

def main():
    # Target properties to predict
    targets = ['band_gap', 'formation_energy_per_atom']  # Add more targets as needed
    
    for target in targets:
        print(f"\nTraining ResNet50 model for {target}...")
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data(target)
        
        # Train model
        model, train_losses, test_losses = train_resnet50(
            X_train, X_test, y_train, y_test,
            target=target,
            batch_size=32,
            num_epochs=5,
            learning_rate=0.001
        )
        
        # Plot training history
        plot_training_history(train_losses, test_losses, target)
        
        # Make predictions
        y_pred = predict_resnet50(model, X_test)
        
        # Calculate and print metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        print(f"\nResults for {target}:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

if __name__ == "__main__":
    main() 