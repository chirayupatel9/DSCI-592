Loaded 9834 records from materials_for_ml.csv

==================================================
Training models to predict band_gap
==================================================

Preprocessing data...
Dataset shape after removing rows with missing band_gap: (9834, 102)
Using 102 features for modeling
Training set shape: (7867, 102)
Test set shape: (1967, 102)

Training Random Forest model...
Random Forest - Best parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
Random Forest - RMSE: 0.8738
Random Forest - MAE: 0.5487
Random Forest - R²: 0.7517

Training Gradient Boosting model...
Gradient Boosting - Best parameters: {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200}
Gradient Boosting - RMSE: 0.8372
Gradient Boosting - MAE: 0.5393
Gradient Boosting - R²: 0.7721

Training Elastic Net model...
Elastic Net - Best parameters: {'alpha': 0.001, 'l1_ratio': 0.1, 'max_iter': 1000}
Elastic Net - RMSE: 1.1442
Elastic Net - MAE: 0.8599
Elastic Net - R²: 0.5742

Comparing model performance...
Best model for band_gap prediction: Gradient Boosting (R² = 0.7721)

==================================================
Training models to predict formation_energy_per_atom
==================================================

Preprocessing data...
Dataset shape after removing rows with missing formation_energy_per_atom: (9834, 102)
Using 101 features for modeling
Training set shape: (7867, 101)
Test set shape: (1967, 101)

Training Random Forest model...
Random Forest - Best parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
Random Forest - RMSE: 0.2777
Random Forest - MAE: 0.1571
Random Forest - R²: 0.9430

Training Gradient Boosting model...
Gradient Boosting - Best parameters: {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}
Gradient Boosting - RMSE: 0.2587
Gradient Boosting - MAE: 0.1497
Gradient Boosting - R²: 0.9505

Training Elastic Net model...
Elastic Net - Best parameters: {'alpha': 0.001, 'l1_ratio': 0.1, 'max_iter': 1000}
Elastic Net - RMSE: 0.4600
Elastic Net - MAE: 0.3188
Elastic Net - R²: 0.8436

Comparing model performance...
Best model for formation_energy_per_atom prediction: Gradient Boosting (R² = 0.9505)