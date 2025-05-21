import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_dir = r"D:\CSE-475Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset"
batch_size = 32
num_epochs = 10
num_classes = 9
lr = 0.0001

class_names = ['Bike', 'Bus', 'Car', 'Cng', 'Cycle', 'Mini-Truck', 'People', 'Rickshaw', 'Truck']
label_map = {name: idx for idx, name in enumerate(class_names)}

transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_map=None):
        self.image_label_pairs = self._get_image_paths_and_labels(root_dir, label_map)
        self.transform = transform

    def _get_image_paths_and_labels(self, root_dir, label_map):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_paths = []
        for ext in exts:
            image_paths.extend(glob(os.path.join(root_dir, '**', ext), recursive=True))
        image_label_pairs = []

        for path in image_paths:
            parent_folder = os.path.basename(os.path.dirname(path))
            if parent_folder in label_map:
                label = label_map[parent_folder]
                image_label_pairs.append((path, label))
        return image_label_pairs

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        img_path, label = self.image_label_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


image_datasets = {
    x: CustomImageDataset(os.path.join(data_dir, f"{x}/images"), transform[x], label_map)
    for x in ['train', 'val']
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
    for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

def train_model(model, criterion, optimizer, num_epochs=10):
    model.to(device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model

models_to_train = {
    "resnet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    "resnet101": models.resnet101(weights=models.ResNet101_Weights.DEFAULT),
    "resnet152": models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
}

for name, model in models_to_train.items():
    print(f"\nTraining {name.upper()} (Full Fine-Tuning)")

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    model = train_model(model, criterion, optimizer, num_epochs=num_epochs)
    end = time.time()

    torch.save(model.state_dict(), f"{name}_finetuned.pth")
    print(f" {name.upper()} training complete in {(end - start) / 60:.2f} minutes")
