Image Input (PNG) ──► CNN (e.g., ResNet18) ─┐
                                            ├──► Fusion ───► Dense layers ───► Prediction
Metadata Input (from JSON) ─► MLP (Dense) ──┘


Model Arch
[Image (224x224)] ─► [CNN: ResNet18] ─► Image Features (Vector)
[Metadata JSON] ─► [MLP: Dense Layers] ─► Metadata Features (Vector)
                   ↓
          [Concatenation]
                   ↓
       [Fusion Dense Layers]
                   ↓
             [Final Output]

- CNN: Extracts patterns, symmetry, texture from image.

- MLP: Extracts useful scalar features (density, bandgap, etc.).

- Fusion Layer: Combines both representations.

- Prediction Head: Regression (if predicting bandgap/energy) or classification (if predicting stability)

Important Feature | Reason
density | Related to structural tightness
band_gap | Target if you want regression
formation_energy_per_atom | Target or feature
energy_above_hull | Stability indicator
is_stable | Binary target
symmetry.crystal_system | Useful for symmetry-aware learning
volume, density_atomic | Size/compactness features
elements (one-hot encode) | Chemical nature (example: Se)