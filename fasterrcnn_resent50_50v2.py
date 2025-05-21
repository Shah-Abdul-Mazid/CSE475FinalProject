import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from tqdm import tqdm
import contextlib
import io
import matplotlib.pyplot as plt
import numpy as np

try:
    from torchvision.models.detection import (
        fasterrcnn_resnet101_fpn, FasterRCNN_ResNet101_FPN_Weights,
        fasterrcnn_resnet101_fpn_v2, FasterRCNN_ResNet101_FPN_V2_Weights,
    )
except ImportError:
    print("Warning: ResNet101 FPN models not available. Removing from backbones list.")

train_img_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\train\images"
val_img_dir = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\split_dataset\val\images"
train_ann_file = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\annotations\train_annotations_coco.json"
val_ann_file = r"D:\CSE475_Project\Bd_Traffic_Dataset_v6\Bangladeshi Traffic Flow Dataset\Bangladeshi Traffic Flow Dataset\annotations\val_annotations_coco.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

output_dir = "D:/CSE475_Project/FasterR-CNN/output"
os.makedirs(output_dir, exist_ok=True)

class CocoDetectionTransform(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
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

transform = T.Compose([T.ToTensor()])

train_dataset = CocoDetectionTransform(train_img_dir, train_ann_file, transform)
val_dataset = CocoDetectionTransform(val_img_dir, val_ann_file, transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

def get_model_by_backbone(backbone_name, num_classes):
    if backbone_name == "resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    elif backbone_name == "resnet101_fpn" and 'fasterrcnn_resnet101_fpn' in globals():
        model = fasterrcnn_resnet101_fpn(weights=FasterRCNN_ResNet101_FPN_Weights.DEFAULT)
    elif backbone_name == "resnet50_fpn_v2":
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    elif backbone_name == "resnet101_fpn_v2" and 'fasterrcnn_resnet101_fpn_v2' in globals():
        model = fasterrcnn_resnet101_fpn_v2(weights=FasterRCNN_ResNet101_FPN_V2_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown or unsupported backbone: {backbone_name}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)

def plot_loss(train_losses, val_losses, path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    print(f"Saved loss plot to {path}")

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

# === Training settings ===
num_classes = 10
backbones = ["resnet50_fpn", "resnet101_fpn", "resnet50_fpn_v2", "resnet101_fpn_v2"]
num_epochs = 50

results = {}

for backbone_name in backbones:
    print(f"\n=== Training with backbone: {backbone_name} ===")
    try:
        model = get_model_by_backbone(backbone_name, num_classes)
    except ValueError as e:
        print(e)
        continue

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    metric_map = MeanAveragePrecision(iou_type='bbox', box_format='xyxy').to(device)

    best_loss = float('inf')
    train_losses, val_losses, val_maps = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_loop = tqdm(train_loader, desc=f"Training Epoch [{epoch+1}/{num_epochs}] {backbone_name}", leave=False)

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

        train_losses.append(total_train_loss)

        # === Validation Phase ===
        total_val_loss = 0
        metric_map.reset()
        has_valid_preds = False

        val_loop = tqdm(val_loader, desc=f"Validating Epoch [{epoch+1}/{num_epochs}] {backbone_name}", leave=False)
        with torch.no_grad():
            for images, targets in val_loop:
                images = list(img.to(device) for img in images)
                targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Compute loss with targets (use train mode for loss)
                model.train()
                loss_dict = model(images, targets_device)
                model.eval()
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                    total_val_loss += losses.item()
                    val_loop.set_postfix(val_loss=losses.item())  # Display validation loss
                else:
                    print(f"Warning: loss_dict is not a dictionary, got {type(loss_dict)}")

                # Predict for metric computation
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

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)

        if has_valid_preds:
            map_result = metric_map.compute()
            val_map = map_result["map"].item()
        else:
            map_result = {"map": torch.tensor(-1.0), "map_per_class": torch.zeros(num_classes-1)}
            val_map = -1.0

        val_maps.append(val_map)

        print(f"Epoch {epoch+1} Train Loss: {total_train_loss:.4f} Val Loss: {avg_val_loss:.4f} mAP: {val_map:.4f}")

        if avg_val_loss < best_loss and avg_val_loss > 0:
            best_loss = avg_val_loss
            save_path = os.path.join(output_dir, f"best_fasterrcnn_{backbone_name}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with val loss {best_loss:.4f} to {save_path}")

        lr_scheduler.step()

    print(f"Best validation loss for {backbone_name}: {best_loss:.4f}")
    if 'save_path' in locals():
        print(f"Best model saved at {save_path}")

    final_model_path = os.path.join(output_dir, f"final_fasterrcnn_{backbone_name}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    plot_loss(train_losses, val_losses, os.path.join(output_dir, f"loss_{backbone_name}.png"))
    plot_map_per_class(map_result, os.path.join(output_dir, f"map_per_class_{backbone_name}.png"))
    visualize_predictions(model, val_dataset, device, n=5, threshold=0.5,
                          save_path=os.path.join(output_dir, f"predictions_{backbone_name}.png"))
    compute_confusion_matrix(model, val_loader, device, num_classes,
                             os.path.join(output_dir, f"confusion_matrix_{backbone_name}.png"))

    results[backbone_name] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_maps": val_maps,
        "best_model_path": save_path if 'save_path' in locals() else None,
        "final_model_path": final_model_path,
    }

print("\n=== Training Complete for all backbones ===")

summary_path = os.path.join(output_dir, "training_summary.txt")
with open(summary_path, "w") as f:
    for backbone_name, result in results.items():
        f.write(f"\n=== Backbone: {backbone_name} ===\n")
        f.write(f"Best Model Path: {result['best_model_path']}\n")
        f.write(f"Final Model Path: {result['final_model_path']}\n")
        f.write(f"Train Losses: {result['train_losses']}\n")
        f.write(f"Validation Losses: {result['val_losses']}\n")
        f.write(f"Validation mAPs: {result['val_maps']}\n")
print(f"Saved training summary to {summary_path}")