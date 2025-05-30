import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow oneDNN warnings
import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from tqdm import tqdm
import contextlib
import io
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Paths to JSON annotation files
train_ann_file = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\train.json"
test_ann_file = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\test\test.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

output_dir = "D:/CSE475_Project/FasterR-CNN/output"
os.makedirs(output_dir, exist_ok=True)

class CocoDetectionTransform(CocoDetection):
    def __init__(self, ann_file, transforms=None):
        img_folder = os.path.join(os.path.dirname(ann_file), "images")
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        path = path.lstrip('train/images/').lstrip('val/images/').lstrip('test/images/').lstrip('images/').lstrip('/')
        full_path = os.path.join(self.root, path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found at: {full_path}")
        return Image.open(full_path).convert("RGB")

    def __getitem__(self, idx):
        idx = int(idx)
        img, target = super().__getitem__(idx)
        if len(target) == 0:
            return self.__getitem__((idx + 1) % len(self))

        boxes, labels, area, iscrowd = [], [], [], []
        for obj in target:
            x, y, w, h = obj['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(obj['category_id'])
            area.append(obj['area'])
            iscrowd.append(obj['iscrowd'])

        target_dict = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        if self._transforms:
            img = self._transforms(img)

        return img, target_dict

# Define transforms
train_transform = T.Compose([
    T.ToTensor(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])
val_transform = T.Compose([T.ToTensor()])

# Create datasets
train_dataset = CocoDetectionTransform(train_ann_file, train_transform)
test_dataset = CocoDetectionTransform(test_ann_file, val_transform) if os.path.exists(test_ann_file) else None

# Create train-validation split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x))) if test_dataset else None

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.box_head.dropout = torch.nn.Dropout(p=0.3)
    return model.to(device)

def plot_loss(train_losses, val_losses, path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    print(f"Saved loss plot to {path}")
    if len(train_losses) > 5 and train_losses[-1] < val_losses[-1] * 0.5:
        print("Warning: Potential overfitting detected")
    if len(train_losses) > 5 and train_losses[-1] > 1.0 and val_losses[-1] > 1.0:
        print("Warning: Potential underfitting detected")

def plot_map_per_class(metric, path):
    aps = metric["map_per_class"].cpu().numpy()
    classes = [f"Class {i+1}" for i in range(len(aps))]
    plt.figure(figsize=(12, 6))
    plt.bar(classes, aps)
    plt.xlabel("Class")
    plt.ylabel("Average Precision (AP)")
    plt.title("mAP per Class")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.savefig(path)
    plt.close()
    print(f"Saved mAP per class plot to {path}")

def visualize_predictions(model, dataset, device, n=5, threshold=0.5, save_path=None):
    model.eval()
    plt.figure(figsize=(15, 10))
    for i in range(n):
        img, target = dataset[i]
        with torch.no_grad():
            prediction = model([img.to(device)])[0]

        img_np = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()
        plt.subplot(1, n, i+1)
        plt.imshow(img_np)
        ax = plt.gca()

        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < threshold:
                continue
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor='red', linewidth=2))
            ax.text(x1, y1, f"{label}: {score:.2f}", fontsize=8,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved prediction visualization to {save_path}")

def compute_confusion_matrix(model, dataloader, device, num_classes, save_path):
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for output, target in zip(outputs, targets):
                preds = output['labels'][output['scores'] > 0.5].cpu().numpy()
                true = target['labels'].cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(true)

    labels_list = list(range(1, num_classes))
    cm = confusion_matrix(all_targets, all_preds, labels=labels_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Class {i}" for i in labels_list])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def evaluate_test_set(model, test_loader, device, num_classes, output_dir):
    if test_loader is None:
        print("No test set provided.")
        return None
    
    model.eval()
    metric_map = MeanAveragePrecision(iou_type='bbox', box_format='xyxy').to(device)
    metric_map.reset()
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating Test Set"):
            images = list(img.to(device) for img in images)
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            
            preds, trues = [], []
            for output, target in zip(outputs, targets):
                if len(output["boxes"]) == 0:
                    continue
                preds.append({
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                })
                trues.append({
                    "boxes": target["boxes"],
                    "labels": target["labels"],
                })
            
            if preds and trues:
                metric_map.update(preds, trues)
    
    map_result = metric_map.compute()
    test_map = map_result["map"].item()
    print(f"Test mAP: {test_map:.4f}")
    
    plot_map_per_class(map_result, os.path.join(output_dir, "map_per_class_test.png"))
    compute_confusion_matrix(model, test_loader, device, num_classes,
                            os.path.join(output_dir, "confusion_matrix_test.png"))
    return test_map

# Training settings
num_classes = 10
num_epochs = 100
patience = 5
backbone_name = "resnet50_fpn"

# Initialize model, optimizer, and scheduler
model = get_model(num_classes)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

metric_map = MeanAveragePrecision(iou_type='bbox', box_format='xyxy').to(device)
writer = SummaryWriter(os.path.join(output_dir, "logs"))

# Training loop
best_loss = float('inf')
epochs_no_improve = 0
train_losses, val_losses, val_maps = [], [], []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    train_loop = tqdm(train_loader, desc=f"Training Epoch [{epoch+1}/{num_epochs}]", leave=False)

    for images, targets in train_loop:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_train_loss += losses.item()
        train_loop.set_postfix(loss=losses.item())

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    total_val_loss = 0
    metric_map.reset()
    has_valid_preds = False

    val_loop = tqdm(val_loader, desc=f"Validating Epoch [{epoch+1}/{num_epochs}]", leave=False)
    with torch.no_grad():
        for images, targets in val_loop:
            images = list(img.to(device) for img in images)
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets_device)
            model.eval()
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()
            val_loop.set_postfix(val_loss=losses.item())

            outputs = model(images)

            preds, trues = [], []
            for output, target in zip(outputs, targets):
                if len(output["boxes"]) == 0:
                    continue
                preds.append({
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                })
                trues.append({
                    "boxes": target["boxes"],
                    "labels": target["labels"],
                })

            if preds and trues:
                has_valid_preds = True
                metric_map.update(preds, trues)

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    if has_valid_preds:
        map_result = metric_map.compute()
        val_map = map_result["map"].item()
    else:
        map_result = {"map": torch.tensor(-1.0), "map_per_class": torch.zeros(num_classes-1)}
        val_map = -1.0

    val_maps.append(val_map)

    writer.add_scalar(f"{backbone_name}/Train_Loss", avg_train_loss, epoch)
    writer.add_scalar(f"{backbone_name}/Val_Loss", avg_val_loss, epoch)
    writer.add_scalar(f"{backbone_name}/Val_mAP", val_map, epoch)

    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} mAP: {val_map:.4f}")

    if avg_val_loss < best_loss and avg_val_loss > 0:
        best_loss = avg_val_loss
        save_path = os.path.join(output_dir, f"best_fasterrcnn_{backbone_name}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model with val loss {best_loss:.4f} to {save_path}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    lr_scheduler.step(avg_val_loss)

# Save final model
final_model_path = os.path.join(output_dir, f"final_fasterrcnn_{backbone_name}.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Saved final model to {final_model_path}")

# Plot and visualize results
plot_loss(train_losses, val_losses, os.path.join(output_dir, f"loss_{backbone_name}.png"))
plot_map_per_class(map_result, os.path.join(output_dir, f"map_per_class_{backbone_name}.png"))
visualize_predictions(model, val_subset, device, n=5, threshold=0.5,
                     save_path=os.path.join(output_dir, f"predictions_{backbone_name}.png"))
compute_confusion_matrix(model, val_loader, device, num_classes,
                        os.path.join(output_dir, f"confusion_matrix_{backbone_name}.png"))

# Evaluate on test set
if test_loader:
    print(f"\n=== Evaluating {backbone_name} on Test Set ===")
    model.load_state_dict(torch.load(save_path))
    test_map = evaluate_test_set(model, test_loader, device, num_classes, output_dir)
else:
    test_map = None

writer.close()

# Save training summary
summary_path = os.path.join(output_dir, "training_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Backbone: {backbone_name}\n")
    f.write(f"Train Losses: {train_losses}\n")
    f.write(f"Validation Losses: {val_losses}\n")
    f.write(f"Validation mAPs: {val_maps}\n")
    f.write(f"Test mAP: {test_map if test_map is not None else 'N/A'}\n")
    f.write(f"Best Model Path: {save_path}\n")
    f.write(f"Final Model Path: {final_model_path}\n")
print(f"Saved training summary to {summary_path}")

print("\n=== Training Complete ===")