import torch
import matplotlib.pyplot as plt
from models.model import NeuralClusteringModel
from dataset import get_dataloaders

# Load data
DATA_DIR = "data/imagenet_100"
train_loader, _, num_classes = get_dataloaders(DATA_DIR)

# Load model
model = NeuralClusteringModel(num_clusters=16, num_classes=num_classes)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Settings
NUM_CLUSTERS = 16
IMAGES_PER_CLUSTER = 5

# Store images per cluster
cluster_images = {i: [] for i in range(NUM_CLUSTERS)}

# Blank image for empty spots
blank_img = torch.zeros_like(next(iter(train_loader))[0][0])

# -----------------------------
# Collect images
# -----------------------------
with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)
        _, assignment = model(images)

        # [B, K, C] → [B, K]
        assignment_scores = assignment.mean(dim=2)
        cluster_ids = torch.argmax(assignment_scores, dim=1)

        for i in range(len(images)):
            cid = int(cluster_ids[i].item())

            # Ignore invalid cluster IDs
            if cid >= NUM_CLUSTERS:
                continue

            if len(cluster_images[cid]) < IMAGES_PER_CLUSTER:
                cluster_images[cid].append(images[i].cpu())

        # Stop only when all clusters have enough images
        if all(len(v) >= IMAGES_PER_CLUSTER for v in cluster_images.values()):
            break

# Fill missing spots with blank images
for i in range(NUM_CLUSTERS):
    while len(cluster_images[i]) < IMAGES_PER_CLUSTER:
        cluster_images[i].append(blank_img)

# -----------------------------
# Cluster labels (change manually if you want semantic names)
# -----------------------------
cluster_labels = [f"Cluster {i}" for i in range(NUM_CLUSTERS)]

# -----------------------------
# Plot (CLEAN GRID WITH LABELS)
# -----------------------------
fig, axes = plt.subplots(NUM_CLUSTERS, IMAGES_PER_CLUSTER, figsize=(8, 26))

for i in range(NUM_CLUSTERS):
    for j in range(IMAGES_PER_CLUSTER):
        axes[i, j].axis("off")

        img = cluster_images[i][j]
        img = img.detach().cpu()
        img = img.permute(1, 2, 0).numpy()
        img = (img + 1) / 2  # normalize [-1,1] → [0,1]
        img = img.clip(0, 1)
        axes[i, j].imshow(img)

    # Add row label on the left of each cluster
    axes[i, 0].text(-0.3, 0.5, f"Cluster {i}",
                transform=axes[i, 0].transAxes,
                fontsize=9, va='center', ha='right')

# Adjust spacing
plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.2)
plt.show()