import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ======================
# LOAD IMAGE
# ======================
image_path = "test.jpg"   # <-- change your image path

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image)

# ======================
# CREATE PATCHES (4x4)
# ======================
patch_size = 4
patches = []

for i in range(0, 224, patch_size):
    for j in range(0, 224, patch_size):
        patch = image_tensor[:, i:i+patch_size, j:j+patch_size]
        patches.append(patch)

patches = torch.stack(patches)

# ======================
# LOAD MODEL
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

# ======================
# EXTRACT PATCH FEATURES
# ======================
features = []

with torch.no_grad():
    for patch in patches:
        patch = patch.unsqueeze(0).to(device)

        # Resize patch to 224x224 for ResNet
        patch = torch.nn.functional.interpolate(patch, size=(224, 224))

        feat = model(patch)
        feat = feat.view(-1)

        features.append(feat.cpu().numpy())

features = np.array(features)

# ======================
# NORMALIZE
# ======================
features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

# ======================
# CLUSTER PATCHES
# ======================
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(features)

print("✅ Patch clustering done!")

# ======================
# REBUILD CLUSTER MAP
# ======================
cluster_map = labels.reshape(224 // patch_size, 224 // patch_size)

# ======================
# VISUALIZE
# ======================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cluster_map, cmap="viridis")
plt.title("Clustered Segments (FEC Style)")
plt.axis("off")

plt.show()