import tempfile
import streamlit as st
from streamlit_option_menu import option_menu
import os
import random
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
import time
from pathlib import Path
import sys
import asyncio
import platform
import warnings
import io
import base64
from datetime import datetime

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import scale_cam_image, show_cam_on_image

# Suppress specific warnings
sys.modules['torch._classes'] = None
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

warnings.filterwarnings("ignore", 
    message="Thread 'MainThread': missing ScriptRunContext",
    module="streamlit.runtime.scriptrunner_utils")

if sys.platform == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

try:
    from streamlit.watcher import local_sources_watcher
    local_sources_watcher.get_module_paths = lambda _: []
except Exception:
    pass

# Class mapping for object detection
class_map = {
    0.0: 'Bike',
    1.0: 'Bus',
    2.0: 'Car',
    3.0: 'Cng',
    4.0: 'Cycle',
    5.0: 'Mini-Truck',
    6.0: 'People',
    7.0: 'Rickshaw',
    8.0: 'Truck'
}

# Define base directory
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
if not BASE_DIR.exists():
    st.error(f"Base directory not found: {BASE_DIR}")
    st.stop()
print(f"Base Directory: {BASE_DIR}")

def grad_cam_and_save(model_path, img_path, save_dir, target_layers_indices=[-2, -3, -4], use_multi_layer=True, file_prefix=""):
    """Generate and save Grad-CAM visualization for an image."""
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = YOLO(model_path)
        model.eval()

        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = np.float32(rgb_img) / 255.0

        if use_multi_layer:
            target_layers = [model.model.model[i] for i in target_layers_indices]
        else:
            target_layers = [model.model.model[target_layers_indices[0]]]

        cam = EigenCAM(model, target_layers, task='od')
        grayscale_cam = cam(rgb_img)[0, :, :]
        cam_image = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{file_prefix}_gradcam.jpg')
        cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, cam_bgr)
        print(f"Saved: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error processing {img_path} with model {model_path}: {e}")
        return None

# Model paths
MODEL_PATHS = {
    "YOLO10_with_SGD": BASE_DIR / "yolo_training" / "yolov10_SGD" / "weights" / "best.pt",
    "YOLO10_with_AdamW": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "weights" / "best.pt",
    "YOLO10_with_Adamax": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "weights" / "best.pt",
    "YOLO10_with_Adam": BASE_DIR / "yolo_training" / "yolov10_Adam" / "weights" / "best.pt",
    "YOLO12_with_SGD": BASE_DIR / "yolo_training" / "yolo12_SGD" / "weights" / "best.pt",
    "YOLO12_with_AdamW": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "weights" / "best.pt",
    "YOLO12_with_Adamax": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "weights" / "best.pt",
    "YOLO12_with_Adam": BASE_DIR / "yolo_training" / "yolo12_Adam" / "weights" / "best.pt",
}
csv_paths = {
    "YOLO10_with_SGD": BASE_DIR / "yolo_training" / "yolov10_SGD" / "overall_metrics.csv",
    "YOLO10_with_AdamW": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "overall_metrics.csv",
    "YOLO10_with_Adamax": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "overall_metrics.csv",
    "YOLO10_with_Adam": BASE_DIR / "yolo_training" / "yolov10_Adam" / "overall_metrics.csv",
    "YOLO12_with_SGD": BASE_DIR / "yolo_training" / "yolo12_SGD" / "overall_metrics.csv",
    "YOLO12_with_AdamW": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "overall_metrics.csv",
    "YOLO12_with_Adamax": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "overall_metrics.csv",
    "YOLO12_with_Adam": BASE_DIR / "yolo_training" / "yolo12_Adam" / "overall_metrics.csv"
}
# Image paths for evaluation plots
IMAGE_PATHS_MAP = {
    "YOLO10_with_SGD": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolov10_SGD" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolov10_SGD" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolov10_SGD" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolov10_SGD" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolov10_SGD" / "R_curve.png",
        "Results" : BASE_DIR / "yolo_training" / "yolov10_SGD" / "results.png"
    },
    "YOLO10_with_Adam": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolov10_Adam" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolov10_Adam" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolov10_Adam" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolov10_Adam" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolov10_Adam" / "R_curve.png",
        "Results" : BASE_DIR / "yolo_training" / "yolov10_Adam" / "results.png"
    },
    "YOLO10_with_Adamax": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "R_curve.png",
        "Result": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "results.png",

    },
    "YOLO10_with_AdamW": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "R_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "results.png"    
    },
    "YOLO12_with_SGD": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolo12_SGD" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolo12_SGD" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolo12_SGD" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolo12_SGD" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_SGD" / "R_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_SGD" / "results.png",
    },
    "YOLO12_with_AdamW": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "R_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "results.png",
    },
    "YOLO12_with_Adamax": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "R_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "results.png",
    },
    "YOLO12_with_Adam": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolo12_Adam" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolo12_Adam" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolo12_Adam" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolo12_Adam" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_Adam" / "R_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_Adam" / "results.png",
    },
}

def get_device():
    """Return the appropriate device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_path):
    """Load a YOLO model from the given path."""
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def run_inference(model, img):
    """Run object detection inference on an image."""
    try:
        if isinstance(img, np.ndarray):
            return model(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        return model(img)
    except Exception as e:
        st.error(f"Inference failed: {str(e)}")
        return None

def draw_boxes_on_image(image, results, class_map=None):
    """Draw bounding boxes and labels on the image."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20) if os.path.exists("arial.ttf") else ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    if results:
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0].item()
                    cls_id = box.cls[0].item()
                    class_name = class_map.get(cls_id, f"Class {cls_id}") if class_map else f"Class {cls_id}"
                    label = f"{class_name} {conf:.2f}"
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                    draw.text((x1, y1 - 20), label, fill="white", font=font)
    return image

def display_images_grid(title, image_paths_dict, cols_per_row=3):
    """Display images in a grid layout."""
    st.subheader(title)
    captions = list(image_paths_dict.keys())
    paths = list(image_paths_dict.values())
    num_images = len(paths)
    
    for i in range(0, num_images, cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, col in enumerate(cols):
            img_idx = i + idx
            if img_idx < num_images:
                path = paths[img_idx]
                caption = captions[img_idx]
                if os.path.exists(path):
                    img = Image.open(path)
                    col.image(img, caption=caption, use_container_width=True)
                else:
                    col.warning(f"Image not found: {path}")

def validate_best_model(model_path, device, data_yaml, project_dir, name):
    """Validate a YOLO model and return metrics."""
    if not os.path.exists(model_path):
        st.error(f"No model found at {model_path}, skipping validation...")
        return None

    try:
        model = YOLO(model_path)
        result = model.val(
            data=data_yaml,
            imgsz=512,
            batch=32,
            device=device,
            plots=True,
            save_json=False,
            project=project_dir,
            name=name,
            exist_ok=True
        )

        metrics = result.box
        class_names = result.names

        data = []
        ap50_values = []
        ap_values = []
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
            ap_values.append(ap)
            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score

            data.append({
                'Class ID': cls_idx,
                'Class Name': cls_name,
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1-Score': f1_score * 100,
                'mAP@0.5': ap50 * 100,
                'mAP@0.5:0.95': ap * 100
            })

        map50 = round(sum(ap50_values) / len(ap50_values), 3) * 100 if ap50_values else 0
        map_50_95 = round(sum(ap_values) / len(ap_values), 3) * 100 if ap_values else 0
        avg_precision = round(total_precision / total_classes, 3) * 100
        avg_recall = round(total_recall / total_classes, 3) * 100
        avg_f1_score = round(total_f1_score / total_classes, 3) * 100

        data.append({
            'Class ID': 'Overall',
            'Class Name': 'Overall Metrics',
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-Score': avg_f1_score,
            'mAP@0.5': map50,
            'mAP@0.5:0.95': map_50_95
        })

        return pd.DataFrame(data)

    except Exception as e:
        st.error(f"Error validating model: {str(e)}")
        return None

def real_time_inference(model, device, video_source=0):
    """Perform real-time object detection using webcam."""
    if "stop_inference" not in st.session_state:
        st.session_state.stop_inference = False

    if st.button("Stop Inference"):
        st.session_state.stop_inference = True

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Error opening video stream. Try a different video source index.")
        return

    placeholder = st.empty()
    
    try:
        while not st.session_state.stop_inference:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))  # Consider making size configurable
            results = run_inference(model, frame)
            if results:
                img_annotated = draw_boxes_on_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), results, class_map)
                frame = np.array(img_annotated)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                placeholder.image(frame, caption="Real-time Object Detection", channels="BGR", use_container_width=True)
            time.sleep(0.03)  # Adjust for smoother performance
    finally:
        cap.release()

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Machine Learning-MRAR",
        page_icon=str(BASE_DIR / "assets" / "artificial-intelligence.png"),
        layout="wide"
    )
    
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Dataset", "Model", "Results", "Real-time Detection", "Upload Video"],
            icons=["house", "table", "gear", "bar-chart", "camera", "film"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Home":
        st.title("Welcome to the Bangladeshi Traffic Flow Object Detection Project on Matchine Learning  🚗")
        
        st.markdown("""
        ## Project Overview
        This project harnesses advanced **YOLOv10**, **YOLOv12**, and **Faster-RCNN_resnet50_fpn** models to detect and classify vehicles in **Bangladeshi traffic flow data**. By analyzing real-world traffic scenarios, we aim to improve traffic monitoring, urban planning, and road safety across Bangladesh.

        ---

        ## Objectives
        - **Vehicle Detection & Classification**: Identify vehicle types (e.g., cars, buses, trucks, rickshaws) in images and videos from Bangladeshi roads.
        - **Traffic Pattern Analysis**: Provide insights into traffic density and vehicle distribution for smart city initiatives.
        - **Model Interpretability**: Use Grad-CAM to visualize regions influencing model predictions.
        - **User-Friendly Interface**: Enable easy image uploads and real-time inference with clear visualizations.

        ---

        ## Bangladeshi Traffic Flow Dataset
        The dataset comprises images and videos from traffic cameras across Bangladesh, annotated in **COCO format**. Key features include:
        - **Diverse Scenarios**: Urban intersections, rural roads, and highways under varied lighting and weather conditions.
        - **Vehicle Classes**: Cars, buses, trucks, motorcycles, rickshaws, and more, capturing Bangladesh’s unique traffic mix.
        - **Challenges**: Overcrowded scenes, occlusions, and low visibility, ideal for testing robust object detection.

        ---

        ## Models
        We employ three state-of-the-art object detection models, fine-tuned on the Bangladeshi traffic dataset:
        - **YOLOv10**: Lightweight and fast, optimized for real-time detection with balanced accuracy.
        - **YOLOv12**: Advanced model with superior feature extraction, excelling in complex scenes.
        - **Faster-RCNN_resnet50_fpn**: Two-stage model offering high accuracy, especially in challenging scenarios.

        ---

        ## Key Features
        ### Grad-CAM Visualization
        **Grad-CAM** (Gradient-weighted Class Activation Mapping) highlights image regions driving model predictions, offering insights into how models interpret traffic scenes.

        ### Real-Time Inference
        Run object detection on uploaded images or live webcam streams, with options to select YOLOv10, YOLOv12, or Faster-RCNN_resnet50_fpn.

        ### Interactive User Interface
        - **Model Selection**: Choose YOLOv10, YOLOv12, or Faster-RCNN_resnet50_fpn for inference.
        - **Image Upload**: Upload images for detection and Grad-CAM visualization.
        - **Real-Time Detection**: Use webcam feeds for live analysis.
        - **Results Visualization**: View annotated images with bounding boxes, labels, and Grad-CAM heatmaps.
        - **Performance Metrics**: Access precision, recall, and mAP in the Metrics section.
        - **Dataset Preview**: Explore sample dataset images to understand data diversity.

        ---

        ## Get Started
        Navigate using the sidebar:
        - **Model**: Upload images for inference with YOLOv10, YOLOv12, or Faster-RCNN_resnet50_fpn.
        - **Dataset**: Learn about the Bangladeshi traffic dataset.
        - **Metrics**: View model performance metrics.

        **Example Use Case**: Upload a Dhaka traffic image to detect vehicles, analyze density, and visualize model focus with Grad-CAM.

        ---

        ## Conclusion
        This project advances traffic management in Bangladesh by delivering accurate vehicle detection and actionable insights. It supports smart city goals, enhances road safety, and streamlines traffic analysis.

        ## Acknowledgments
        We thank the YOLO and Faster-RCNN research communities, dataset contributors, and the open-source ecosystem for enabling this work.

        ## Future Work
        - **Model Optimization**: Enhance model accuracy and speed through further fine-tuning.
        - **Traffic System Integration**: Partner with authorities to deploy real-time monitoring solutions.
        - **User Feedback**: Incorporate feedback to add features like vehicle counting or speed estimation.
        - **Multi-Modal Analysis**: Combine camera and sensor data for comprehensive traffic insights.<br>
        """,unsafe_allow_html=True)
        st.image(BASE_DIR / "assets" / "poster.jpg", caption="Traffic in Dhaka, Bangladesh", use_container_width=True)
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; margin-top: 30px;'>
                <img src='data:image/png;base64,{developer_img_base64}' width='300'><br>
                <p style='margin-top: 10px; font-size: 16px;'>
                <strong>Developed By</strong>:<br>
                <em>
                <strong>Shah Abdul Mazid</strong>, 
                <strong>Mahia Meherun Safa</strong>, 
                <strong>Abir Sarwar</strong>, and 
                <strong>Raida Sultana</strong><br>
                for smarter traffic solutions in Bangladesh.</em>
                </p>
                <p style='margin-top: 10px; font-size: 14px;'>
                <strong>Developed</strong><br>
                This project was built using Streamlit, PyTorch, and OpenCV, leveraging the power of YOLO and Faster-RCNN frameworks. The application is designed to be scalable and user-friendly, with a focus on real-world applicability in Bangladesh's traffic management systems.
                </p>
            </div>
            """.format(
                developer_img_base64=base64.b64encode(open(BASE_DIR / "assets" / "developer.jpg", "rb").read()).decode(),
            ), unsafe_allow_html=True)
    elif selected == "Dataset":
        st.subheader("Dataset Preview")
        st.write("Preview random images from the Bangladeshi Traffic Flow Dataset.")
        num_images = st.number_input("Number of images to preview:", min_value=1, max_value=100, value=5)
        images_per_row = st.number_input("Images per row:", min_value=1, max_value=10, value=5)

        root_dataset_path = BASE_DIR / "Bd_Traffic_Dataset_v6" / "Bangladeshi Traffic Flow Dataset" / "Bangladeshi Traffic Flow Dataset" / "Raw Images"
        if os.path.exists(root_dataset_path):
            all_image_paths = []
            for root, _, files in os.walk(root_dataset_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_image_paths.append(os.path.join(root, file))

            if all_image_paths:
                samples = random.sample(all_image_paths, min(num_images, len(all_image_paths)))
                for i in range(0, len(samples), images_per_row):
                    cols = st.columns(images_per_row)
                    for j, img_path in enumerate(samples[i:i+images_per_row]):
                        try:
                            image = Image.open(img_path)
                            cols[j].image(image, caption=os.path.basename(img_path), use_container_width=True)
                        except:
                            cols[j].warning(f"Failed to load image: {img_path}")
            else:
                st.warning("No image files found.")
        else:
            st.error(f"Dataset not found at: {root_dataset_path}")
            
    elif selected == "Model":
        st.subheader("Run Inference on Uploaded Image")
        st.write("Upload a JPG, PNG, or JPEG image to run object detection and Grad-CAM visualization using the selected YOLO model. All outputs will be saved in the same folder.")

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        model_choice = st.selectbox("Select YOLO Model", ["select a model"] + list(MODEL_PATHS.keys()))
        folder_name = st.text_input("Output folder name (optional, defaults to timestamp)", "")

        if uploaded_file and model_choice != "select a model":
            try:
                image = Image.open(uploaded_file).convert("RGB")
                model_path = MODEL_PATHS.get(model_choice)
                model = get_model(model_path)

                # Define output directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = folder_name.strip() or timestamp
                save_dir = str(BASE_DIR / "yolo_grad_cam_output" / folder_name)
                os.makedirs(save_dir, exist_ok=True)

                # Generate unique file prefix
                file_prefix = f"image_{timestamp}"

                # Save input image
                input_image_path = os.path.join(save_dir, f"{file_prefix}_input.jpg")
                image.save(input_image_path, format="JPEG")
                st.write(f"Input image saved: {input_image_path}")

                if model:
                    results = run_inference(model, image)
                    if results:
                        # Generate and save object detection output
                        img_annotated = draw_boxes_on_image(image.copy(), results, class_map)
                        detection_image_path = os.path.join(save_dir, f"{file_prefix}_detection.jpg")
                        img_annotated.save(detection_image_path, format="JPEG")
                        st.write(f"Detection output saved: {detection_image_path}")

                        # Run Grad-CAM
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            image.save(tmp_file.name)
                            temp_img_path = tmp_file.name

                        try:
                            gradcam_path = grad_cam_and_save(
                                model_path=model_path,
                                img_path=temp_img_path,
                                save_dir=save_dir,
                                target_layers_indices=[-2, -3],
                                use_multi_layer=True,
                                file_prefix=file_prefix
                            )
                            st.write(f"Grad-CAM output saved: {gradcam_path}")
                            
                            # Resize images for display
                            def resize_image(image, target_width):
                                aspect_ratio = image.height / image.width
                                target_height = int(target_width * aspect_ratio)
                                return image.resize((target_width, target_height))

                            input_image = resize_image(Image.open(input_image_path), target_width=300)
                            annotated_image = resize_image(Image.open(detection_image_path), target_width=300)
                            gradcam_image = resize_image(Image.open(gradcam_path), target_width=300) if gradcam_path and os.path.exists(gradcam_path) else None

                            # Display results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.image(input_image, caption="Input Image", use_container_width=True)
                            with col2:
                                st.image(annotated_image, caption="Detection Output", use_container_width=True)
                            with col3:
                                if gradcam_image:
                                    st.image(gradcam_image, caption="Grad-CAM Visualization", use_container_width=True)
                                else:
                                    st.warning("Grad-CAM generation failed.")
                        finally:
                            if os.path.exists(temp_img_path):
                                os.unlink(temp_img_path)
                    else:
                        st.error("No detection results.")
                else:
                    st.error("Model could not be loaded.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please upload an image and select a model.")

    elif selected == "Results":
        st.subheader("Evaluation Results")
        st.write("View class-wise and overall metrics for each YOLO model, along with evaluation plots.")
        metrics_cols = ['Precision', 'Recall', 'F1-Score', 'mAP@0.5', 'mAP@0.5:0.95']
        data_yaml = BASE_DIR / "Bd_Traffic_Dataset_v6" / "Bangladeshi Traffic Flow Dataset" / "Bangladeshi Traffic Flow Dataset" / "data.yaml"
        if not data_yaml.exists():
            st.error(f"Data YAML file not found: {data_yaml}")
            st.stop()
        device = get_device()

        for idx, (model_key, model_path) in enumerate(MODEL_PATHS.items()):
            model_name = model_key.replace("YOLO10_with_", "YOLOv10 ").replace("YOLO12_with_", "YOLOv12 ").replace("_", " ")
            st.subheader(f"{model_name} Validation Results")
            image_paths = IMAGE_PATHS_MAP.get(model_key, {})
            name = model_key.lower().replace("yolo10_with_", "yolov10_").replace("yolo12_with_", "yolov12_")
            with st.spinner(f"Validating {model_name} model ({idx + 1}/{len(MODEL_PATHS)})..."):
                df_metrics = validate_best_model(
                    model_path=model_path,
                    device=device,
                    data_yaml=data_yaml,
                    project_dir="yolo_training",
                    name=name
                )
            if df_metrics is not None:
                st.markdown("### Class-wise and Overall Metrics")
                for col in metrics_cols:
                    df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce')
                format_dict = {col: "{:.2f}%" for col in metrics_cols if pd.api.types.is_numeric_dtype(df_metrics[col])}
                st.dataframe(
                    df_metrics.style.format(format_dict),
                    use_container_width=True
                )

            st.subheader(f"{model_name} Confusion Matrix and Curves")
            if image_paths:
                display_images_grid(f"{model_name} Plots", image_paths)
            else:
                st.info(f"No plots available for {model_name}. Please check the plot directory.")

    elif selected == "Real-time Detection":
        st.subheader("Real-time Object Detection")
        st.write("Perform object detection using your webcam. Click 'Stop Inference' to end the session.")
        model_choice_rt = st.selectbox("Select YOLO Model for Real-time Detection", ["select a model"] + list(MODEL_PATHS.keys()))

        if model_choice_rt != "select a model":
            model_path = MODEL_PATHS.get(model_choice_rt)
            model = get_model(model_path)
            if model:
                st.info("Starting webcam inference. Click 'Stop Inference' to stop.")
                real_time_inference(model, get_device())
            else:
                st.error("Model could not be loaded.")

    elif selected == "Upload Video":
        st.subheader("Upload a Video for Inference")
        st.write("Upload an MP4, AVI, or MOV video to run object detection using the selected YOLO model.")
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        model_choice_vid = st.selectbox("Select YOLO Model for Video Inference", ["select a model"] + list(MODEL_PATHS.keys()))

        if video_file is not None and model_choice_vid != "select a model":
            model_path = MODEL_PATHS.get(model_choice_vid)
            model = get_model(model_path)
            if not model:
                st.error("Model could not be loaded.")
                return

            input_video_path = None
            output_video_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(video_file.read())
                    input_video_path = tfile.name

                st.info("Processing video...")
                cap = cv2.VideoCapture(input_video_path)
                if not cap.isOpened():
                    st.error("Error opening video file.")
                    return

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

                frame_count = 0
                with st.spinner(f"Processing {total_frames} frames..."):
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        results = run_inference(model, frame)
                        if results:
                            img_annotated = draw_boxes_on_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), results, class_map)
                            frame_out = cv2.cvtColor(np.array(img_annotated), cv2.COLOR_RGB2BGR)
                            out.write(frame_out)
                        frame_count += 1
                        if frame_count % 50 == 0:
                            st.text(f"Processed {frame_count}/{total_frames} frames")

                cap.release()
                out.release()

                st.success("Video processing complete!")
                st.video(output_video_path)

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
            finally:
                if input_video_path and os.path.exists(input_video_path):
                    os.unlink(input_video_path)
                if output_video_path and os.path.exists(output_video_path):
                    os.unlink(output_video_path)

if __name__ == "__main__":
    main()
    