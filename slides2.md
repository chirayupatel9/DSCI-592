
---
title: DINOv2 + UMAP + t-SNE Embedding Pipeline
layout: cover
---

# 🔍 Exploring Image Embeddings  
### Using DINOv2, UMAP, PCA & t-SNE  
Chirayu Patel

---

# ⚙️ Setup & Model

- Using `timm` to load `ViT-small` pretrained with DINOv2
- Device: CUDA-enabled GPU (if available)
- Model scripted to TorchScript for deployment

```python
dino_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
dino_model.eval().to(device)
```

---

# 📦 Data Processing

- Images loaded via `OpenCV` and resized to 32x32
- Normalized & transformed to 224x224 before passing to ViT
- Feature extraction from CLS token

```python
img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), ...)
features = dino_model.forward_features(batch_tensors)[:, 0]
```

---

# 📊 Feature Embeddings

- Total images: `{{ features_np.shape[0] }}`
- Feature vector shape: `(384,)`

```python
features = extract_features_dino(images_np)
```

---

# 📉 UMAP Visualization

```python
umap.UMAP(n_components=2).fit_transform(features)
```

![](https://via.placeholder.com/600x400?text=UMAP+Output)

---

# 📈 PCA Visualization

- Fast linear projection
- Often used before t-SNE for speed

```python
PCA(n_components=2).fit_transform(features)
```

![](https://via.placeholder.com/600x400?text=PCA+Output)

---

# 🧠 t-SNE (Slow but Powerful)

- Perplexity: 30
- 1000 iterations

```python
TSNE(n_components=2).fit_transform(features_pca_50)
```

![](https://via.placeholder.com/600x400?text=tSNE+Output)

---

# 🤖 Parametric UMAP (Keras)

- Feedforward MLP encoder
- GPU accelerated with TensorFlow

```python
ParametricUMAP(encoder=keras_encoder).fit_transform(features_np)
```

![](https://via.placeholder.com/600x400?text=Parametric+UMAP)

---

# 🧠 t-SNE Approximation

- Learned mapping from features to 2D t-SNE space
- Train simple MLP on `(features, tsne_output)`

```python
nn.Sequential(
  nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 2)
)
```

---

# 📊 t-SNE Approximation Plot

![](https://via.placeholder.com/600x400?text=tSNE+NN+Output)

---

# 💾 Model Export

- Scripted feature extractor for deployment

```python
torch.jit.script(DinoFeatureExtractor(dino_model))
```

Saved to: `dino_model/1/model.pt`

---

# 📚 Summary

- ✅ Extracted features using DINOv2
- ✅ Visualized using UMAP, PCA, and t-SNE
- ✅ Trained NN to approximate t-SNE
- ✅ Created deployable TorchScript model

---

# 🙌 Thank You!

> Questions?  
Let's discuss further applications (e.g., clustering, search, anomaly detection)

---

layout: center
class: text-center
---

# 🎉 End of Presentation  
Made with ❤️ by Chirayu Patel  
