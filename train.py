import torch
import torch.nn as nn
import torch.optim as optim
from models.model import NeuralClusteringModel
from dataset import get_dataloaders
import torch.nn.functional as F


# Path to dataset
DATA_DIR = "data/imagenet_100"

# Load data
train_loader, val_loader, num_classes = get_dataloaders(DATA_DIR)

# Model
model = NeuralClusteringModel(num_clusters=16, num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

# Loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}")

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs, assignment = model(images)
        # ✅ Classification loss
        classification_loss = criterion(outputs, labels)

        # ✅ Convert assignment → cluster probabilities
        # assignment shape: [B, K, H, W] OR similar
        p = assignment.mean(dim=2)   # ⭐ IMPORTANT
        # 🔥 Sharpen distribution (VERY IMPORTANT)
        p = p ** 2
        p = p / (p.sum(dim=1, keepdim=True) + 1e-8)

        # ✅ ENTROPY LOSS (forces clear clusters)
        entropy_loss = - (p * torch.log(p + 1e-8)).sum(dim=1).mean()

        # ✅ BALANCE LOSS (prevents single cluster collapse)
        cluster_mean = p.mean(dim=0)
        balance_loss = (cluster_mean * torch.log(cluster_mean + 1e-8)).sum()

        # ✅ FINAL LOSS
        loss = classification_loss + 1.0 * entropy_loss + 1.0 * balance_loss
        print(f"Cls: {classification_loss.item():.3f}, Ent: {entropy_loss.item():.3f}, Bal: {balance_loss.item():.3f}")
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("Training Complete!")

# Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved!")


