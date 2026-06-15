import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.data import Subset


def get_dataloaders(data_dir, batch_size=16):

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=transform
    )
    
    import torch
    from torch.utils.data import Subset

    train_indices = torch.randperm(len(train_dataset))[:500]
    val_indices = torch.randperm(len(val_dataset))[:200]

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, len(train_dataset.dataset.classes)