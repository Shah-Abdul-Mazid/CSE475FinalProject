import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from transformers import DetrForObjectDetection

# Paths
train_ann_file = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\annotations\train_annotations_coco.json"
val_ann_file = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\annotations\val_annotations_coco.json"
train_img_folder = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\images"
val_img_folder = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\val\images"

# Validate paths
for path in [train_img_folder, val_img_folder, train_ann_file, val_ann_file]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

# Hyperparameters
num_classes = 9
batch_size = 32
num_epochs = 50
lr = 1e-4
weight_decay = 1e-4
backbone_lr = 1e-3  # Learning rate for SGD (backbone)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = T.Compose([
    T.ToTensor(),
    T.Resize((800, 800)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom COCO dataset
class FlattenedCoco(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile, transforms)
        self.root = root

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = [{k: torch.tensor(v, dtype=torch.int64) if isinstance(v, list) else v for k, v in t.items()} for t in [target]]
        return img, target[0]

# Datasets and DataLoaders
train_dataset = FlattenedCoco(train_img_folder, train_ann_file, transform=transform)
val_dataset = FlattenedCoco(val_img_folder, val_ann_file, transform=transform)

collate_fn = lambda x: tuple(zip(*x))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model setup


model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
in_features = model.class_labels_classifier.in_features 
model.class_labels_classifier = torch.nn.Linear(in_features, num_classes + 1)

# model = "facebook/detr-resnet-50"
# in_features = model.class_embed.in_features
# model.class_embed = torch.nn.Linear(in_features, num_classes + 1)  # +1 for background
# model.to(device)

# Split parameters
backbone_params = list(model.backbone.parameters())
transformer_params = list(model.transformer.parameters())
head_params = list(model.class_embed.parameters()) + list(model.bbox_embed.parameters())

# Define optimizers
backbone_optimizer = SGD(backbone_params, lr=backbone_lr, momentum=0.9, weight_decay=weight_decay)
transformer_optimizer = Adam(transformer_params, lr=lr, weight_decay=weight_decay)
head_optimizer = AdamW(head_params, lr=lr, weight_decay=weight_decay)

# Define schedulers
backbone_scheduler = StepLR(backbone_optimizer, step_size=10, gamma=0.1)
transformer_scheduler = StepLR(transformer_optimizer, step_size=10, gamma=0.1)
head_scheduler = StepLR(head_optimizer, step_size=10, gamma=0.1)

# Setup for saving and tracking
save_dir = "detr_model_checkpoints"
os.makedirs(save_dir, exist_ok=True)

loss_history = {
    "train_total": [], "train_loss_ce": [], "train_loss_bbox": [], "train_loss_giou": [],
    "val_total": [], "val_loss_ce": [], "val_loss_bbox": [], "val_loss_giou": []
}
best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_losses = {"total": 0, "ce": 0, "bbox": 0, "giou": 0}
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for images, targets in train_pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero gradients for all optimizers
        backbone_optimizer.zero_grad()
        transformer_optimizer.zero_grad()
        head_optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        # Backward pass
        loss.backward()

        # Update parameters
        backbone_optimizer.step()
        transformer_optimizer.step()
        head_optimizer.step()

        train_losses["total"] += loss.item()
        train_losses["ce"] += loss_dict.get("loss_ce", 0).item()
        train_losses["bbox"] += loss_dict.get("loss_bbox", 0).item()
        train_losses["giou"] += loss_dict.get("loss_giou", 0).item()

        train_pbar.set_postfix({
            "Total": f"{loss.item():.4f}",
            "CE": f"{loss_dict.get('loss_ce', 0).item():.4f}",
            "BBox": f"{loss_dict.get('loss_bbox', 0).item():.4f}",
            "GIoU": f"{loss_dict.get('loss_giou', 0).item():.4f}"
        })

    # Average losses
    for key in train_losses:
        train_losses[key] /= len(train_loader)
        loss_history[f"train_loss_{key}" if key != "total" else "train_total"].append(train_losses[key])

    # Step schedulers
    backbone_scheduler.step()
    transformer_scheduler.step()
    head_scheduler.step()

    # Validation loop
    model.eval()
    val_losses = {"total": 0, "ce": 0, "bbox": 0, "giou": 0}
    
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for images, targets in val_pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            val_losses["total"] += loss.item()
            val_losses["ce"] += loss_dict.get("loss_ce", 0).item()
            val_losses["bbox"] += loss_dict.get("loss_bbox", 0).item()
            val_losses["giou"] += loss_dict.get("loss_giou", 0).item()

            val_pbar.set_postfix({
                "Total": f"{loss.item():.4f}",
                "CE": f"{loss_dict.get('loss_ce', 0).item():.4f}",
                "BBox": f"{loss_dict.get('loss_bbox', 0).item():.4f}",
                "GIoU": f"{loss_dict.get('loss_giou', 0).item():.4f}"
            })

    # Average validation losses
    for key in val_losses:
        val_losses[key] /= len(val_loader)
        loss_history[f"val_loss_{key}" if key != "total" else "val_total"].append(val_losses[key])

    # Save best model
    if val_losses["total"] < best_val_loss:
        best_val_loss = val_losses["total"]
        torch.save({
            'model_state_dict': model.state_dict(),
            'backbone_optimizer': backbone_optimizer.state_dict(),
            'transformer_optimizer': transformer_optimizer.state_dict(),
            'head_optimizer': head_optimizer.state_dict(),
            'epoch': epoch
        }, os.path.join(save_dir, "detr_best_model.pth"))

    # Save periodic checkpoints
    if (epoch + 1) % 5 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'backbone_optimizer': backbone_optimizer.state_dict(),
            'transformer_optimizer': transformer_optimizer.state_dict(),
            'head_optimizer': head_optimizer.state_dict(),
            'epoch': epoch
        }, os.path.join(save_dir, f"detr_epoch_{epoch+1}.pth"))

    # Print epoch results
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train - Total: {train_losses['total']:.4f} | CE: {train_losses['ce']:.4f} | BBox: {train_losses['bbox']:.4f} | GIoU: {train_losses['giou']:.4f}")
    print(f"Val   - Total: {val_losses['total']:.4f} | CE: {val_losses['ce']:.4f} | BBox: {val_losses['bbox']:.4f} | GIoU: {val_losses['giou']:.4f}")
    print("-" * 60)

# Plotting loss curves
plt.figure(figsize=(10, 7))
epochs_range = range(1, num_epochs + 1)
plt.plot(epochs_range, loss_history["train_total"], label="Train Total")
plt.plot(epochs_range, loss_history["val_total"], label="Val Total")
plt.plot(epochs_range, loss_history["train_loss_ce"], label="Train CE")
plt.plot(epochs_range, loss_history["val_loss_ce"], label="Val CE")
plt.plot(epochs_range, loss_history["train_loss_bbox"], label="Train BBox")
plt.plot(epochs_range, loss_history["val_loss_bbox"], label="Val BBox")
plt.plot(epochs_range, loss_history["train_loss_giou"], label="Train GIoU")
plt.plot(epochs_range, loss_history["val_loss_giou"], label="Val GIoU")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DETR Training & Validation Losses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "detr_loss_curves.png"))

# Save loss history to CSV
df_loss = pd.DataFrame({
    "epoch": list(range(1, num_epochs + 1)),
    "train_total_loss": loss_history["train_total"],
    "train_loss_ce": loss_history["train_loss_ce"],
    "train_loss_bbox": loss_history["train_loss_bbox"],
    "train_loss_giou": loss_history["train_loss_giou"],
    "val_total_loss": loss_history["val_total"],
    "val_loss_ce": loss_history["val_loss_ce"],
    "val_loss_bbox": loss_history["val_loss_bbox"],
    "val_loss_giou": loss_history["val_loss_giou"]
})
df_loss.to_csv(os.path.join(save_dir, "loss_history.csv"), index=False)
print(f"✅ Loss curves and CSV saved in: {save_dir}")