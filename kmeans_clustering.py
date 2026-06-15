import torch
import torchvision.models as models
from torchvision.models import resnet101, ResNet101_Weights
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataset import get_dataloaders
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import defaultdict

# ======================
# SETTINGS
# ======================
MAX_IMAGES = 1000   # 🔥 IMPORTANT (improves clustering)
NUM_CLUSTERS = 8
IMAGES_PER_CLUSTER = 5

# ======================
# LOAD DATA
# ======================
train_loader, _, _ = get_dataloaders("data/imagenet_100")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ======================
# MODEL (ResNet101)
# ======================
model = resnet101(weights=ResNet101_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model.to(device)
model.eval()

features_list = []
images_list = []

# ======================
# FEATURE EXTRACTION (LIMITED DATA)
# ======================
count = 0

with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)

        features = model(images)
        features_list.append(features.cpu().numpy())
        images_list.append(images.cpu())

        count += images.size(0)
        if count >= MAX_IMAGES:
            break

# ======================
# PREPARE FEATURES
# ======================
features = np.concatenate(features_list, axis=0)

# Standardize
scaler = StandardScaler()
features = scaler.fit_transform(features)

# PCA (better config)
pca = PCA(n_components=25, whiten=True)
features = pca.fit_transform(features)

# Normalize
features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

# ======================
# CLUSTERING
# ======================
kmeans = MiniBatchKMeans(
    n_clusters=NUM_CLUSTERS,
    random_state=42,
    batch_size=512,
    n_init=20
)

labels = kmeans.fit_predict(features)

print("✅ Clustering done!")


# ======================
# VISUALIZATION (IMPROVED)
# ======================
images_all = torch.cat(images_list, dim=0)

cluster_items = defaultdict(list)

# Store (index, feature)
for i, cid in enumerate(labels):
    cluster_items[cid].append((i, features[i]))

selected_images = {}

for cid, items in cluster_items.items():
    center = kmeans.cluster_centers_[cid]

    # Sort by distance to cluster center (BEST REPRESENTATIVES)
    items_sorted = sorted(
        items,
        key=lambda x: np.linalg.norm(x[1] - center)
    )

    selected_images[cid] = [
        images_all[i] for i, _ in items_sorted[:IMAGES_PER_CLUSTER]
    ]

sorted_clusters = sorted(selected_images.keys())

plt.figure(figsize=(20, 14))

for row, cid in enumerate(sorted_clusters):
    imgs = selected_images[cid]

    for col in range(IMAGES_PER_CLUSTER):
        plt.subplot(len(sorted_clusters), IMAGES_PER_CLUSTER, row * IMAGES_PER_CLUSTER + col + 1)

        if col < len(imgs):
            img = imgs[col]
            img = img.permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            plt.imshow(img)

        plt.axis("off")

        if col == 0:
            plt.ylabel(f"Cluster {cid}", fontsize=9)

plt.suptitle("Final Clustering (Best Representatives)", fontsize=16)
plt.tight_layout()
plt.show()

# ======================
# EVALUATION
# ======================
score = silhouette_score(features, labels)
print(f"⭐ Silhouette Score: {score:.4f}")