"""
Train machine learning models to predict material properties.
This script uses the prepared data to build predictive models for band gap and other properties.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

def load_data():
    """Load the prepared data from CSV"""
    try:
        df = pd.read_csv("materials_for_ml.csv")
        print(f"Loaded {len(df)} records from materials_for_ml.csv")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df, target='band_gap'):
    """Preprocess data for machine learning"""
    if df is None or df.empty:
        print("No data available for preprocessing")
        return None, None, None, None
    
    print("\nPreprocessing data...")
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target])
    print(f"Dataset shape after removing rows with missing {target}: {df.shape}")
    
    # Select features
    # Categorical features to convert to one-hot encoding
    categorical_cols = ["crystal_system"]
    
    # Identify element presence columns
    element_cols = [col for col in df.columns if col.startswith('contains_')]
    
    # Select numerical features
    numerical_cols = [
        "nelements", "nsites", "density", "volume", 
        "formation_energy_per_atom", "energy_per_atom"
    ]
    
    # Remove target from numerical features if it's there
    if target in numerical_cols:
        numerical_cols.remove(target)
    
    # For crystal systems, convert to one-hot encoding
    if "crystal_system" in df.columns:
        # Ensure crystal_system is a string type
        df["crystal_system"] = df["crystal_system"].astype(str)
        # Convert to one-hot encoding
        crystal_system_dummies = pd.get_dummies(df["crystal_system"], prefix="crystal")
        # Drop the original crystal_system column
        df = df.drop("crystal_system", axis=1)
        # Concatenate the one-hot encoded columns
        df = pd.concat([df, crystal_system_dummies], axis=1)
    
    # Combine feature sets
    crystal_system_cols = [col for col in df.columns if col.startswith('crystal_')]
    
    # Combine all features
    feature_cols = numerical_cols + element_cols + crystal_system_cols
    print(f"Using {len(feature_cols)} features for modeling")
    
    # Split data into features and target
    X = df[feature_cols]
    y = df[target]
    
    # Ensure all features are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any remaining NaN values with 0
    X = X.fillna(0)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_train = X_train[numerical_cols].copy()
    numerical_test = X_test[numerical_cols].copy()
    
    X_train[numerical_cols] = scaler.fit_transform(numerical_train)
    X_test[numerical_cols] = scaler.transform(numerical_test)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, X_test, y_train, y_test, target='band_gap'):
    """Train and evaluate a Random Forest model"""
    print("\nTraining Random Forest model...")
    
    # Parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    # Initialize model
    rf = RandomForestRegressor(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Fit model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_rf.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest - Best parameters: {grid_search.best_params_}")
    print(f"Random Forest - RMSE: {rmse:.4f}")
    print(f"Random Forest - MAE: {mae:.4f}")
    print(f"Random Forest - R²: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Random Forest: Actual vs Predicted {target}')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.tight_layout()
    plt.savefig(f'plots/rf_{target}_actual_vs_predicted.png')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
    plt.title(f'Random Forest: Top 15 Feature Importance for {target}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'plots/rf_{target}_feature_importance.png')
    
    # Save the model
    joblib.dump(best_rf, f'models/random_forest_{target}.joblib')
    
    return best_rf, feature_importance, rmse, r2

def train_gradient_boosting(X_train, X_test, y_train, y_test, target='band_gap'):
    """Train and evaluate a Gradient Boosting model"""
    print("\nTraining Gradient Boosting model...")
    
    # Parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    # Initialize model
    gb = GradientBoostingRegressor(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Fit model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_gb = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_gb.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Gradient Boosting - Best parameters: {grid_search.best_params_}")
    print(f"Gradient Boosting - RMSE: {rmse:.4f}")
    print(f"Gradient Boosting - MAE: {mae:.4f}")
    print(f"Gradient Boosting - R²: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Gradient Boosting: Actual vs Predicted {target}')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.tight_layout()
    plt.savefig(f'plots/gb_{target}_actual_vs_predicted.png')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_gb.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
    plt.title(f'Gradient Boosting: Top 15 Feature Importance for {target}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'plots/gb_{target}_feature_importance.png')
    
    # Save the model
    joblib.dump(best_gb, f'models/gradient_boosting_{target}.joblib')
    
    return best_gb, feature_importance, rmse, r2

def train_elastic_net(X_train, X_test, y_train, y_test, target='band_gap'):
    """Train and evaluate an Elastic Net model"""
    print("\nTraining Elastic Net model...")
    
    # Parameter grid for grid search
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9],
        'max_iter': [1000, 2000]
    }
    
    # Initialize model
    en = ElasticNet(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=en,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Fit model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_en = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_en.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Elastic Net - Best parameters: {grid_search.best_params_}")
    print(f"Elastic Net - RMSE: {rmse:.4f}")
    print(f"Elastic Net - MAE: {mae:.4f}")
    print(f"Elastic Net - R²: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Elastic Net: Actual vs Predicted {target}')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.tight_layout()
    plt.savefig(f'plots/en_{target}_actual_vs_predicted.png')
    
    # Feature coefficients
    coef = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': best_en.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_features = coef.iloc[:15]
    plt.barh(top_features['Feature'], top_features['Coefficient'])
    plt.title(f'Elastic Net: Top 15 Feature Coefficients for {target}')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'plots/en_{target}_coefficients.png')
    
    # Save the model
    joblib.dump(best_en, f'models/elastic_net_{target}.joblib')
    
    return best_en, coef, rmse, r2

def compare_models(results, target):
    """Compare the performance of different models"""
    print("\nComparing model performance...")
    
    models = list(results.keys())
    rmse_vals = [results[model]['rmse'] for model in models]
    r2_vals = [results[model]['r2'] for model in models]
    
    # Plot RMSE comparison
    plt.figure(figsize=(10, 6))
    plt.bar(models, rmse_vals, color=['blue', 'green', 'orange'])
    plt.title(f'Model Comparison: RMSE for {target}')
    plt.ylabel('RMSE (lower is better)')
    plt.tight_layout()
    plt.savefig(f'plots/model_comparison_rmse_{target}.png')
    
    # Plot R² comparison
    plt.figure(figsize=(10, 6))
    plt.bar(models, r2_vals, color=['blue', 'green', 'orange'])
    plt.title(f'Model Comparison: R² for {target}')
    plt.ylabel('R² (higher is better)')
    plt.tight_layout()
    plt.savefig(f'plots/model_comparison_r2_{target}.png')
    
    # Print best model
    best_idx = np.argmax(r2_vals)
    best_model = models[best_idx]
    print(f"Best model for {target} prediction: {best_model} (R² = {r2_vals[best_idx]:.4f})")
    
    return best_model

def main():
    """Main function to train and evaluate models"""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Targets to predict
    targets = ['band_gap', 'formation_energy_per_atom']
    
    for target in targets:
        print(f"\n{'='*50}")
        print(f"Training models to predict {target}")
        print(f"{'='*50}")
        
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(df, target=target)
        if X_train is None:
            continue
        
        # Train and evaluate models
        rf_model, rf_importance, rf_rmse, rf_r2 = train_random_forest(X_train, X_test, y_train, y_test, target)
        gb_model, gb_importance, gb_rmse, gb_r2 = train_gradient_boosting(X_train, X_test, y_train, y_test, target)
        en_model, en_coef, en_rmse, en_r2 = train_elastic_net(X_train, X_test, y_train, y_test, target)
        
        # Compare models
        results = {
            'Random Forest': {'rmse': rf_rmse, 'r2': rf_r2},
            'Gradient Boosting': {'rmse': gb_rmse, 'r2': gb_r2},
            'Elastic Net': {'rmse': en_rmse, 'r2': en_r2}
        }
        
        best_model = compare_models(results, target)
        
        # Save the feature importance of the best model
        if best_model == 'Random Forest':
            rf_importance.to_csv(f'models/best_feature_importance_{target}.csv', index=False)
        elif best_model == 'Gradient Boosting':
            gb_importance.to_csv(f'models/best_feature_importance_{target}.csv', index=False)
        else:
            en_coef.to_csv(f'models/best_feature_importance_{target}.csv', index=False)

if __name__ == "__main__":
    main() 