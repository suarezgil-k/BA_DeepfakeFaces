import torch
from torchvision import datasets, transforms
import clip
from tqdm import tqdm
import os

# Pfad anpassen
save_dir = "../data/Embeddings"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# CLIP-Modell laden
model, preprocess = clip.load("ViT-L/14", device=device)

# CIFAR10 laden
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

datasets_dict = {
    "train": datasets.CIFAR10(root="../data/Datasets", train=True, download=True, transform=transform),
    "val": datasets.CIFAR10(root="../data/Datasets", train=False, download=True, transform=transform),
    "test": datasets.CIFAR10(root="../data/Datasets", train=False, download=False, transform=transform),
}

with torch.no_grad():
    for split_name, dataset in datasets_dict.items():
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        all_features, all_labels = [], []
        for images, labels in tqdm(loader, desc=f"Processing {split_name}"):
            images = images.to(device)
            features = model.encode_image(images)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())
            all_labels.append(labels)
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)

        out_path = os.path.join(save_dir, f"images_cifar10_{split_name}_CLIP-ViT-L14.torch")
        torch.save({"features": all_features, "labels": all_labels}, out_path)
        print(f"Saved {split_name} embeddings to {out_path}")
