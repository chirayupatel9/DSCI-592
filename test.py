import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from cuml.manifold import TSNE
import cupy as cp
import time

def generate_tsne_from_images(
    model_path,
    image_folder,
    n_classes=17,
    batch_size=512,
    num_workers=16,
    output_dim=2,
    perplexity=30,
    truncate_fc_at=6,
    plot=True,
    device_str="cuda:0"
):
    start_time = time.time()
    # Set device
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ---- Define custom model ----
    def resnet50_(in_channels, n_classes, dropout=0.5, weights=None):
        model = models.resnet50(weights=weights)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(512, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout),
            nn.Linear(64, n_classes, bias=True)
        )
        return model

    # ---- Load model and weights ----
    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    model = resnet50_(in_channels=3, n_classes=n_classes)
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Truncate FC head (for embeddings)
    model.fc = nn.Sequential(*list(model.fc.children())[:truncate_fc_at])

    print(f"✅ Model loaded on {device}")

    # ---- Custom Dataset ----
    class UnlabeledImageDataset(Dataset):
        def __init__(self, folder_path, transform=None):
            self.image_paths = sorted([
                os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")
            ])
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, self.image_paths[idx]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = UnlabeledImageDataset(image_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    # ---- Feature extraction ----
    embeddings = []
    file_names = []

    with torch.no_grad():
        for inputs, paths in tqdm(loader, desc="Extracting Features"):
            inputs = inputs.to(device, non_blocking=True)
            feats = model(inputs)
            embeddings.append(feats.cpu().numpy())
            file_names.extend(paths)

    embeddings = np.vstack(embeddings)

    # ---- cuML t-SNE ----
    print("🔁 Running cuML t-SNE...")
    embeddings_gpu = cp.asarray(embeddings)
    tsne = TSNE(n_components=output_dim, perplexity=perplexity, n_iter=1000, verbose=1)
    tsne_result_gpu = tsne.fit_transform(embeddings_gpu)
    tsne_result = cp.asnumpy(tsne_result_gpu)

    # ---- Plot ----
    if plot:
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=10)
        plt.title("cuML t-SNE Projection")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    end_time = time.time()
    print(f"🔁 cuML t-SNE completed in {end_time - start_time:.2f} seconds")
    
    return tsne_result, file_names

tsne_coords, image_paths = generate_tsne_from_images(
    model_path="yichen_model.pth",
    image_folder="data/output/png",
    device_str="cuda:4"
)