# %% [markdown]
# # Materials Science Machine Learning Project
# 
# ## Project Timeline (8 Weeks)
# 
# ### Week 1-2: Data Exploration and Preprocessing
# - Perform comprehensive EDA on materials science dataset
# - Clean and preprocess data
# - Handle missing values and outliers
# - Feature engineering and selection
# 
# ### Week 3-4: Model Development
# - Implement and compare multiple ML algorithms:
#   - Random Forest
#   - Gradient Boosting
#   - Neural Networks
#   - Support Vector Machines
# - Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
# - Cross-validation and model evaluation
# 
# ### Week 5-6: Model Optimization
# - Feature importance analysis
# - Ensemble methods
# - Advanced techniques (e.g., stacking, boosting)
# - Model interpretability using SHAP/LIME
# 
# ### Week 7: API Development
# - Develop Flask/FastAPI for model deployment
# - Create prediction endpoints
# - Implement data validation
# - Add documentation
# 
# ### Week 8: Finalization and Deployment
# - Model deployment
# - Performance monitoring
# - Documentation completion
# - Project presentation
# 
# ## Project Goals
# 1. Develop a high-accuracy predictive model for materials properties
# 2. Create an interpretable model that provides insights into material characteristics
# 3. Build a robust API for real-time predictions
# 4. Document the entire process and findings
# 
# ## Success Metrics
# - Model accuracy > 90%
# - Robust API with error handling
# - Comprehensive documentation
# - Clear visualization of results
# 
# ## Deliverables
# 1. Clean, preprocessed dataset
# 2. Trained and optimized ML model
# 3. API for predictions
# 4. Documentation and presentation
# 5. Code repository with all scripts

# %% [markdown]
# # Materials Science Dataset EDA
# 
# This script performs exploratory data analysis on the materials science dataset to understand its characteristics, distributions, and relationships between features.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
from pymongo import MongoClient
from dbconfig import MONGO_URI, MONGO_DB, MONGO_JSON_COLLECTION

# Set plotting style
plt.style.use('default')  # Using default style instead of seaborn
sns.set_theme()  # This will set the seaborn theme
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# %% [markdown]
# ## 2. Data Loading and Initial Overview

# %%
# Load data from CSV
df = pd.read_csv('materials_for_ml.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
display(df.head())

print("\nData types:")
display(df.dtypes)

print("\nBasic statistics:")
display(df.describe())

# %% [markdown]
# ## 3. Missing Value Analysis

# %%
# Calculate missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})

print("Missing Value Analysis:")
display(missing_data[missing_data['Missing Values'] > 0])

# Visualize missing values
plt.figure(figsize=(12, 6))
msno.matrix(df)
plt.title('Missing Values Matrix')
plt.show()

# %% [markdown]
# ## 4. Target Variable Analysis (Band Gap)

# %%
# Distribution plot
plt.figure(figsize=(12, 6))
sns.histplot(df['band_gap'], bins=50, kde=True)
plt.title('Distribution of Band Gap')
plt.xlabel('Band Gap (eV)')
plt.ylabel('Count')
plt.show()

# Box plot
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['band_gap'])
plt.title('Box Plot of Band Gap')
plt.show()

# Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(df['band_gap'].dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot of Band Gap')
plt.show()

# %% [markdown]
# ## 5. Feature Distributions

# %%
# Select numerical features
numerical_features = ['nelements', 'nsites', 'density', 'volume', 'formation_energy_per_atom', 'energy_per_atom']

# Plot distributions for each numerical feature
for feature in numerical_features:
    plt.figure(figsize=(12, 6))
    sns.histplot(df[feature], bins=50, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()
    
    # Box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.show()

# %% [markdown]
# ## 6. Correlation Analysis

# %%
# Calculate correlation matrix
correlation_matrix = df[numerical_features + ['band_gap']].corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Plot correlation with band gap
target_correlations = correlation_matrix['band_gap'].sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=target_correlations.index, y=target_correlations.values)
plt.title('Correlation with Band Gap')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Crystal System Analysis

# %%
# Crystal system distribution
plt.figure(figsize=(12, 6))
df['crystal_system'].value_counts().plot(kind='bar')
plt.title('Distribution of Crystal Systems')
plt.xlabel('Crystal System')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Band gap by crystal system
plt.figure(figsize=(12, 6))
sns.boxplot(x='crystal_system', y='band_gap', data=df)
plt.title('Band Gap by Crystal System')
plt.xlabel('Crystal System')
plt.ylabel('Band Gap (eV)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Element Analysis

# %%
# Count elements in materials
element_columns = [col for col in df.columns if col.startswith('contains_')]
element_counts = df[element_columns].sum().sort_values(ascending=False)

# Plot top 20 elements
plt.figure(figsize=(12, 6))
element_counts.head(20).plot(kind='bar')
plt.title('Top 20 Most Common Elements')
plt.xlabel('Element')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Average band gap by element presence
element_band_gaps = {}
for element in element_columns:
    element_band_gaps[element] = df[df[element]]['band_gap'].mean()

element_band_gaps = pd.Series(element_band_gaps).sort_values(ascending=False)

# Plot top 20 elements by average band gap
plt.figure(figsize=(12, 6))
element_band_gaps.head(20).plot(kind='bar')
plt.title('Top 20 Elements by Average Band Gap')
plt.xlabel('Element')
plt.ylabel('Average Band Gap (eV)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Outlier Analysis

# %%
# Function to detect outliers using IQR method
def detect_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

# Analyze outliers for each numerical feature
for feature in numerical_features + ['band_gap']:
    outliers = detect_outliers(df, feature)
    if len(outliers) > 0:
        print(f"\nOutliers in {feature}:")
        print(f"Number of outliers: {len(outliers)}")
        print(f"Percentage of outliers: {len(outliers)/len(df)*100:.2f}%")
        
        # Plot box plot with outliers highlighted
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df[feature])
        plt.title(f'Box Plot of {feature} with Outliers')
        plt.show()

# %% [markdown]
# ## 10. Summary and Insights

# %%
# Generate summary statistics
summary_stats = df.describe()

# Calculate skewness and kurtosis
skewness = df[numerical_features + ['band_gap']].skew()
kurtosis = df[numerical_features + ['band_gap']].kurtosis()

print("Summary Statistics:")
display(summary_stats)

print("\nSkewness:")
display(skewness)

print("\nKurtosis:")
display(kurtosis)

# Print key insights
print("\nKey Insights:")
print(f"1. Dataset contains {len(df)} materials with {len(df.columns)} features")
print(f"2. Missing values percentage: {df.isnull().sum().sum()/df.size*100:.2f}%")
print(f"3. Number of numerical features: {len(numerical_features)}")
print(f"4. Number of categorical features: {len(df.select_dtypes(include=['object', 'bool']).columns)}")
print(f"5. Band gap statistics:")
print(f"   - Mean: {df['band_gap'].mean():.2f} eV")
print(f"   - Median: {df['band_gap'].median():.2f} eV")
print(f"   - Standard Deviation: {df['band_gap'].std():.2f} eV")
print(f"   - Range: {df['band_gap'].min():.2f} to {df['band_gap'].max():.2f} eV") 