import torch
from ultralytics import YOLO
import os
import pandas as pd


def get_device():
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("GPU not available, using CPU instead.")
        return torch.device("cpu")

def train_model(model, device, optimizer):
    data_yaml = 'D:/CSE-475Project/Bd_Traffic_Dataset_v6/Bangladeshi Traffic Flow Dataset/Bangladeshi Traffic Flow Dataset/data.yaml'

    epochs = 50
    batch_size = 32
    imgsz = 512
    
    learning_rate = 0.01
    patience = 5

    print(f"\nTraining with {optimizer} optimizer...")

    result = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        optimizer=optimizer,
        lr0=learning_rate,
        device=device,
        project='yolo_training',
        name=f'yolo12_{optimizer}',
        exist_ok=True,
        workers=4,
        val=True,
        plots=True,
        cos_lr=True
    )

def validate_best_model(optimizer, device):
    print(f"\nValidating best model for {optimizer} optimizer...")

    best_model_path = os.path.join('yolo_training', f'yolo12_{optimizer}', 'weights', 'best.pt')
    if not os.path.exists(best_model_path):
        print(f"No best.pt found for {optimizer}, skipping validation...")
        return None

    model = YOLO(best_model_path)
    model.to(device)

    data_yaml = 'D:/CSE-475Project/Bd_Traffic_Dataset_v6/Bangladeshi Traffic Flow Dataset/Bangladeshi Traffic Flow Dataset/data.yaml'
    
    result = model.val(
        data=data_yaml,
        imgsz=512,
        batch=32,
        device=device
    )

    metrics = result.box
    class_names = result.names

    data = []
    ap50_values = []
    
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_classes = len(class_names)
    
    for cls_idx, cls_name in class_names.items():
        precision = round(metrics.p[cls_idx], 3)
        recall = round(metrics.r[cls_idx], 3)
        f1_score = round(metrics.f1[cls_idx], 3)
        ap50 = round(metrics.ap50[cls_idx], 3)
        ap = round(metrics.ap[cls_idx], 3)

        ap50_values.append(ap50)

        total_precision += precision
        total_recall += recall
        total_f1_score += f1_score

        data.append({
            'Class ID': cls_idx,
            'Class Name': cls_name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'AP50': ap50,
            'AP': ap
        })

    map50 = round(sum(ap50_values) / len(ap50_values), 3) if ap50_values else 0

    avg_precision = round(total_precision / total_classes, 3)
    avg_recall = round(total_recall / total_classes, 3)
    avg_f1_score = round(total_f1_score / total_classes, 3)

    data.append({
        'Class ID': 'Overall',
        'Class Name': 'Overall Metrics',
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F1-Score': avg_f1_score,
        'AP50': map50,
        'AP': ''
    })

    csv_path = os.path.join('yolo_training', f'yolo12_{optimizer}', 'overall_metrics.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Overall and classwise metrics saved to {csv_path}")

    validation_result = {
        'Optimizer': optimizer,
        'Precision': round(metrics.mp, 3),
        'Recall': round(metrics.mr, 3),
        'F1-Score': round(2 * (metrics.mp * metrics.mr) / (metrics.mp + metrics.mr), 3) if (metrics.mp + metrics.mr) != 0 else 0,
        'mAP@0.5': round(metrics.map50, 3),
        'mAP@0.5:0.95': round(metrics.map, 3)
    }

    return validation_result

def main():
    print("YOLO11 model loading...")
    validation_results = [] 

    try:
        base_model_path = "D:/CSE-475Project/model/yolo12s.pt"
        base_model = YOLO(base_model_path)
        print("yolo12s model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    device = get_device()
    optimizers = ['SGD', 'Adam', 'Adamax', 'AdamW']

    for optimizer in optimizers:
        best_model_path = os.path.join('yolo_training', f'yolo12_{optimizer}', 'weights', 'best.pt')

        if os.path.exists(best_model_path):
            print(f"\nbest.pt already exists for {optimizer}, skipping training...")
        else:
            try:
                model = YOLO(base_model_path)
                model.to(device)
                train_model(model, device, optimizer)
            except Exception as e:
                print(f"Error training with {optimizer}: {e}")
                continue

        val_result = validate_best_model(optimizer, device)
        if val_result:
            validation_results.append(val_result)

    if validation_results:
        os.makedirs('yolo_validation', exist_ok=True)
        df = pd.DataFrame(validation_results)
        csv_path = 'yolo_validation/best_model_validation_results_yolo12.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nAll validation results saved to {csv_path}")
    else:
        print("No validation results to save.")

if __name__ == '__main__':
    main()
    print("YOLO12 training and validation completed.")