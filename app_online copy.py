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
import logging
from logging.handlers import RotatingFileHandler
import plotly.express as px
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import scale_cam_image, show_cam_on_image

# Setup logging
handler = RotatingFileHandler("app.log", maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings(
    "ignore",
    message="Thread 'MainThread': missing ScriptRunContext",
    module="streamlit.runtime.scriptrunner_utils"
)

# Define base directory
BASE_DIR = Path(__file__).parent

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

# Define base directory and dataset path
root_dataset_path = BASE_DIR / "Raw Image" / "Raw Images"
yolo_training_path = BASE_DIR / "yolo_training"

# Validate base directory
if not BASE_DIR.exists():
    st.error(f"Base directory not found: {BASE_DIR}")
    logger.error(f"Base directory not found: {BASE_DIR}")
    st.stop()
logger.info(f"Base Directory: {BASE_DIR}")

# Validate dataset directory
if not root_dataset_path.exists():
    st.error(f"Dataset directory not found: {root_dataset_path}")
    logger.error(f"Dataset directory not found: {root_dataset_path}")
    st.stop()

# Function to get list of subdirectories (locations) recursively
def get_dataset_locations(root_path):
    """Get list of subdirectory names in the dataset path using os.walk."""
    locations = []
    for dirpath, dirnames, _ in os.walk(root_path):
        for dirname in dirnames:
            locations.append(dirname)
    return sorted(locations) if locations else ["No subdirectories found"]

# Dataset configuration
DATASET_CONFIG = {
    "root_dataset_path": root_dataset_path
}
YOLO_CONFIG = {
    "yolo_training_path": yolo_training_path
}

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

# Validate model paths and filter valid models
valid_models = {name: path for name, path in MODEL_PATHS.items() if path.exists()}
for model_name, model_path in MODEL_PATHS.items():
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
if not valid_models:
    st.error("No valid model files found. Please ensure model weights are in the correct paths.")
    logger.error("No valid model files found in MODEL_PATHS.")
    st.stop()

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
        "Results": BASE_DIR / "yolo_training" / "yolov10_SGD" / "results.png"
    },
    "YOLO10_with_AdamW": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "R_curve.png",
        "Results": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "results.png"
    },
    "YOLO10_with_Adamax": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "R_curve.png",
        "Results": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "results.png"
    },
    "YOLO10_with_Adam": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolov10_Adam" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolov10_Adam" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolov10_Adam" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolov10_Adam" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolov10_Adam" / "R_curve.png",
        "Results": BASE_DIR / "yolo_training" / "yolov10_Adam" / "results.png"
    },
    "YOLO12_with_SGD": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolo12_SGD" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolo12_SGD" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolo12_SGD" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolo12_SGD" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_SGD" / "R_curve.png",
        "Results": BASE_DIR / "yolo_training" / "yolo12_SGD" / "results.png"
    },
    "YOLO12_with_AdamW": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "R_curve.png",
        "Results": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "results.png"
    },
    "YOLO12_with_Adamax": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "R_curve.png",
        "Results": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "results.png"
    },
    "YOLO12_with_Adam": {
        "Normalized Confusion Matrix": BASE_DIR / "yolo_training" / "yolo12_Adam" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR / "yolo_training" / "yolo12_Adam" / "F1_curve.png",
        "Precision Curve": BASE_DIR / "yolo_training" / "yolo12_Adam" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR / "yolo_training" / "yolo12_Adam" / "PR_curve.png",
        "Recall Curve": BASE_DIR / "yolo_training" / "yolo12_Adam" / "R_curve.png",
        "Results": BASE_DIR / "yolo_training" / "yolo12_Adam" / "results.png"
    }
}

def get_model(model_path):
    """Load and return the YOLO model."""
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None

def run_inference(model, image):
    """Run inference on the given image."""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image")
        results = model(image)
        if results is None:
            raise ValueError("Model returned no results")
        return results
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        st.error(f"Inference error: {str(e)}")
        return None

def draw_boxes_on_image(image, results, class_map):
    """Draw dark bounding boxes with larger, clear confidence scores on the image."""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            logger.warning("Arial font not found; falling back to default font.")
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = box.cls.cpu().numpy()[0]
                conf = box.conf.cpu().numpy()[0]
                label = f"{class_map.get(float(cls_id), 'Unknown')} {conf:.2f}"
                draw.rectangle(xyxy, outline="black", width=4)
                text_x = xyxy[0]
                text_y = xyxy[1] - 30 if xyxy[1] - 30 > 0 else xyxy[1] + 30
                bbox = draw.textbbox((text_x, text_y), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                background_coords = [text_x, text_y, text_x + text_width, text_y + text_height]
                draw.rectangle(background_coords, fill=(0, 0, 0, 180))
                draw.text((text_x, text_y), label, fill="white", font=font)
        return image
    except Exception as e:
        logger.error(f"Error drawing boxes: {str(e)}")
        st.error(f"Error drawing boxes: {str(e)}")
        return image

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
        logger.info(f"Saved Grad-CAM: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Error processing Grad-CAM for {img_path} with model {model_path}: {str(e)}")
        return None

def get_device():
    """Return the appropriate device for inference."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def display_images_grid(title, image_paths):
    """Display images in a grid."""
    st.subheader(title)
    for name, path in image_paths.items():
        if path.exists():
            st.image(str(path), caption=name, use_container_width=True)
        else:
            st.warning(f"Image not found: {path}")
            logger.warning(f"Image not found: {path}")

def find_camera():
    """Find an available camera index."""
    for index in range(5):  # Try indices 0 to 4
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
        cap.release()
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
            frame = cv2.resize(frame, (640, 480)) 
            results = run_inference(model, frame)
            if results:
                img_annotated = draw_boxes_on_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), results, class_map)
                frame = np.array(img_annotated)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                placeholder.image(frame, caption="Real-time Object Detection", channels="BGR", use_container_width=True)
            time.sleep(0.03) 
    finally:
        cap.release()

def get_available_codec():
    """Return an available video codec, prioritizing H.264."""
    codecs = ["avc1", "mp4v", "XVID"]
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            temp_out = cv2.VideoWriter("test.mp4", fourcc, 30, (640, 480))
            if temp_out.isOpened():
                temp_out.release()
                os.remove("test.mp4")
                return codec
            temp_out.release()
        except:
            continue
    return None

def main():
    """Main function to run the Streamlit app."""
    icon_path = BASE_DIR / "assets" / "artificial-intelligence.png"
    if not icon_path.exists():
        logger.warning(f"Icon file not found: {icon_path}")
        icon_path = None
    st.set_page_config(
        page_title="Machine Learning-MRAR",
        page_icon=str(icon_path) if icon_path else None,
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
        st.title("Welcome to the Bangladeshi Traffic Flow Object Detection Project 🚗")
        locations = get_dataset_locations(DATASET_CONFIG["root_dataset_path"])
        locations_str = ", ".join(locations) if locations else "Various locations"
        st.markdown(f"""
        ## Project Overview
        This project uses **YOLOv10** and **YOLOv12** models to detect and classify vehicles in **Bangladeshi traffic flow data**. It aims to enhance traffic monitoring, urban planning, and road safety in Bangladesh.

        ## Objectives
        - **Vehicle Detection & Classification**: Identify vehicle types (e.g., cars, buses, trucks, rickshaws).
        - **Traffic Pattern Analysis**: Provide insights into traffic density and vehicle distribution.
        - **Model Interpretability**: Use Grad-CAM to visualize model decision-making.
        - **User-Friendly Interface**: Support image uploads, real-time inference, and visualizations.

        ## Bangladeshi Traffic Flow Dataset
        - **Diverse Scenarios**: Urban intersections, rural roads, highways under varied conditions.
        - **Locations**: {locations_str}
        - **Vehicle Classes**: Cars, buses, trucks, motorcycles, rickshaws, etc.
        - **Challenges**: Overcrowded scenes, occlusions, low visibility.
        - **Total Images**: 23,678

        ## Models
        - **YOLOv10**: Fast, lightweight, suitable for real-time detection.
        - **YOLOv12**: Advanced feature extraction for complex scenes.

        ## Key Features
        - **Grad-CAM Visualization**: Highlights regions influencing predictions.
        - **Real-Time Inference**: Detects objects in webcam streams or uploaded videos.
        - **Interactive UI**: Model selection, image uploads, and performance metrics.

        ## Get Started
        Use the sidebar to:
        - **Model**: Upload images for inference and Grad-CAM.
        - **Dataset**: Preview dataset images.
        - **Results**: View model performance metrics.
        - **Real-time Detection**: Run live detection via webcam.
        - **Upload Video**: Process uploaded videos.

        ## Acknowledgments
        Thanks to the YOLO community, dataset contributors, and open-source ecosystem.

        ## Future Work
        - Optimize models for better accuracy and speed.
        - Integrate with traffic systems for real-time monitoring.
        - Add features like vehicle counting or speed estimation.
        """)
        poster_path = BASE_DIR / "assets" / "poster.jpg"
        if poster_path.exists():
            st.image(poster_path, caption="Traffic in Dhaka, Bangladesh", use_container_width=True)
        else:
            logger.warning(f"Poster image not found: {poster_path}")
        st.markdown("---")
        developer_path = BASE_DIR / "assets" / "developer.jpg"
        if developer_path.exists():
            st.markdown("""
                <div style='text-align: center; margin-top: 30px;'>
                    <img src='data:image/png;base64,{developer_img_base64}' width='300'><br>
                    <p style='margin-top: 10px; font-size: 16px;'>
                    <strong>Developed By</strong>:<br>
                    <em>Shah Abdul Mazid, Mahia Meherun Safa, Abir Sarwar, Raida Sultana</em>
                    </p>
                </div>
                """.format(
                    developer_img_base64=base64.b64encode(open(developer_path, "rb").read()).decode()
                ), unsafe_allow_html=True)
        else:
            logger.warning(f"Developer image not found: {developer_path}")

    elif selected == "Dataset":
        st.subheader("Dataset Preview")
        st.write("Preview random images from the Bangladeshi Traffic Flow Dataset.")
        try:
            num_images = st.number_input("Number of images to preview:", min_value=1, max_value=500, value=5, step=1)
            images_per_row = st.number_input("Images per row:", min_value=1, max_value=10, value=5, step=1)
            logger.info(f"num_images: {num_images}, images_per_row: {images_per_row}")

            @st.cache_data
            def get_image_paths(root_path):
                """Safely get all image paths from directory and subdirectories."""
                image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
                image_paths = []
                for dirpath, _, filenames in os.walk(root_path):
                    for filename in filenames:
                        if filename.lower().endswith(image_extensions):
                            image_paths.append(Path(dirpath) / filename)
                return image_paths

            all_image_paths = get_image_paths(root_dataset_path)
            if all_image_paths:
                num_samples = min(num_images, len(all_image_paths))
                samples = random.sample(all_image_paths, num_samples)
                for i in range(0, len(samples), images_per_row):
                    cols = st.columns(images_per_row)
                    for j, img_path in enumerate(samples[i:i+images_per_row]):
                        try:
                            with Image.open(img_path) as img:
                                cols[j].image(img, caption=img_path.name, use_container_width=True)
                        except Exception as e:
                            cols[j].warning(f"Failed to load {img_path.name}: {str(e)}")
                            logger.error(f"Failed to load image {img_path}: {str(e)}")
            else:
                st.warning(f"No images found in: {root_dataset_path}")
                logger.warning(f"No images found in: {root_dataset_path}")
        except Exception as e:
            st.error(f"Error in Dataset section: {str(e)}")
            logger.error(f"Error in Dataset section: {str(e)}")

    elif selected == "Model":
        st.subheader("Run Inference on Uploaded Image")
        st.write("Upload a JPG, PNG, or JPEG image for object detection and Grad-CAM visualization.")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        model_choice = st.selectbox("Select YOLO Model", ["select a model"] + list(valid_models.keys()))
        folder_name = st.text_input("Output folder name (optional, defaults to timestamp)", "")

        if uploaded_file and model_choice != "select a model":
            try:
                image = Image.open(uploaded_file).convert("RGB")
                model_path = valid_models.get(model_choice)
                model = get_model(model_path)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = folder_name.strip() or timestamp
                save_dir = str(BASE_DIR / "yolo_grad_cam_output" / folder_name)
                os.makedirs(save_dir, exist_ok=True)

                file_prefix = f"image_{timestamp}"
                input_image_path = os.path.join(save_dir, f"{file_prefix}_input.jpg")
                image.save(input_image_path, format="JPEG")
                st.write(f"Input image saved: {input_image_path}")
                logger.info(f"Input image saved: {input_image_path}")

                if model:
                    results = run_inference(model, image)
                    if results:
                        img_annotated = draw_boxes_on_image(image.copy(), results, class_map)
                        detection_image_path = os.path.join(save_dir, f"{file_prefix}_detection.jpg")
                        img_annotated.save(detection_image_path, format="JPEG")
                        st.write(f"Detection output saved: {detection_image_path}")
                        logger.info(f"Detection output saved: {detection_image_path}")

                        with tempfile.TemporaryDirectory() as tmp_dir:
                            temp_img_path = os.path.join(tmp_dir, "temp.jpg")
                            image.save(temp_img_path)
                            gradcam_path = grad_cam_and_save(
                                model_path=model_path,
                                img_path=temp_img_path,
                                save_dir=save_dir,
                                use_multi_layer=True,
                                file_prefix=file_prefix
                            )
                            st.write(f"Grad-CAM output saved: {gradcam_path}")

                            def resize_image(image, target_width):
                                image.thumbnail((target_width, target_width))
                                return image

                            input_image = resize_image(Image.open(input_image_path), target_width=300)
                            annotated_image = resize_image(Image.open(detection_image_path), target_width=300)
                            gradcam_image = resize_image(Image.open(gradcam_path), target_width=300) if gradcam_path and os.path.exists(gradcam_path) else None

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
                    else:
                        st.error("No detection results.")
                else:
                    st.error("Model could not be loaded.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Error in Model section: {str(e)}")
        else:
            st.warning("Please upload an image and select a model.")

    elif selected == "Results":
        st.subheader("Evaluation Results")
        st.write("View class-wise and overall metrics for each YOLO model, along with evaluation plots.")
        metrics_cols = ['Precision', 'Recall', 'F1-Score', 'AP50', 'AP']
        data_yaml = BASE_DIR / "Bd_Traffic_Dataset_v6" / "Bangladeshi Traffic Flow Dataset" / "Bangladeshi Traffic Flow Dataset" / "data.yaml"

        for idx, (model_key, model_path) in enumerate(valid_models.items()):
            model_name = model_key.replace("YOLOv10_with_", "YOLO10 ").replace("YOLO12_with_", "YOLOv12 ").replace("_", " ")
            st.subheader(f"{model_name} Validation Results")
            image_paths = IMAGE_PATHS_MAP.get(model_key, {})
            csv_path = csv_paths.get(model_key)

            if not data_yaml.exists() and csv_path and csv_path.exists():
                st.warning(f"data.yaml not found at {data_yaml}. Reading metrics from {csv_path} instead.")
                logger.warning(f"data.yaml not found at {data_yaml}. Reading metrics from {csv_path} instead.")
                try:
                    df_metrics = pd.read_csv(csv_path)
                    df_metrics = df_metrics.rename(columns={'Class Name': 'Class', 'AP50': 'mAP@0.5', 'AP': 'mAP@0.5:0.95'})
                except Exception as e:
                    st.error(f"Failed to read metrics from {csv_path}: {str(e)}")
                    logger.error(f"Failed to read metrics from {csv_path}: {str(e)}")
                    continue
            elif csv_path and csv_path.exists():
                try:
                    df_metrics = pd.read_csv(csv_path)
                    df_metrics = df_metrics.rename(columns={'Class Name': 'Class', 'AP50': 'mAP@0.5', 'AP': 'mAP@0.5:0.95'})
                except Exception as e:
                    st.error(f"Failed to read metrics from {csv_path}: {str(e)}")
                    logger.error(f"Failed to read metrics from {csv_path}: {str(e)}")
                    continue
            else:
                st.warning(f"No metrics available for {model_name}. Ensure {csv_path} exists.")
                logger.warning(f"No metrics available for {model_name}. Ensure {csv_path} exists.")
                continue

            if df_metrics is not None:
                st.markdown("### Class-wise and Overall Metrics")
                for col in metrics_cols:
                    if col in df_metrics.columns:
                        df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce')
                format_dict = {col: "{:.2f}%" for col in metrics_cols if col in df_metrics.columns and pd.api.types.is_numeric_dtype(df_metrics[col])}
                st.dataframe(
                    df_metrics.style.format(format_dict),
                    use_container_width=True
                )

                df_plot = df_metrics[df_metrics['Class'] != 'Overall Metrics']
                if 'mAP@0.5' in df_plot.columns:
                    st.markdown("### mAP@0.5 Comparison Across Classes")
                    fig = px.bar(df_plot, x="Class", y="mAP@0.5", title=f"{model_name} mAP@0.5 Comparison",
                                 color="Class", color_discrete_sequence=px.colors.qualitative.Plotly)
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"{model_name} Confusion Matrix and Curves")
            if image_paths:
                display_images_grid(f"{model_name} Plots", image_paths)
            else:
                st.info(f"No plots available for {model_name}. Please check the plot directory.")
                logger.info(f"No plots available for {model_name}")

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
        st.write("Upload an MP4, AVI, or MOV video to run object detection using the selected YOLO model. The processed video will be available for download.")
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        model_choice_vid = st.selectbox("Select YOLO Model for Video Inference", ["select a model"] + list(valid_models.keys()))
        
        # Additional options for video processing
        resize_factor = st.slider("Output Video Resize Factor", min_value=0.1, max_value=1.0, value=1.0, step=0.1, help="Reduce resolution to speed up processing (e.g., 0.5 for half size).")
        frame_skip = st.slider("Process Every Nth Frame", min_value=1, max_value=10, value=1, step=1, help="Process every Nth frame to speed up processing (1 = process all frames).")
        output_subfolder = st.text_input("Output Subfolder Name (optional, defaults to timestamp)", "", help="Enter a subfolder name to save the processed video (e.g., 'my_videos'). Saved in output_videos/<subfolder>.")

        if video_file is not None and model_choice_vid != "select a model":
            # Display input video
            st.subheader("Input Video")
            input_video_bytes = video_file.read()
            st.video(input_video_bytes, format=video_file.type)
            logger.info("Input video displayed in Streamlit")

            model_path = valid_models.get(model_choice_vid)
            model = get_model(model_path)
            if not model:
                st.error("Model could not be loaded.")
                logger.error("Model could not be loaded.")
                return

            input_video_path = None
            output_video_path = None
            try:
                # Save input video to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(input_video_bytes)
                    input_video_path = tfile.name
                logger.info(f"Input video saved to: {input_video_path}")

                # Open input video
                cap = cv2.VideoCapture(input_video_path)
                if not cap.isOpened():
                    st.error("Error: Could not open the input video file. Ensure the file is a valid video format (MP4, AVI, MOV) and not corrupted. Try converting the video using FFmpeg (e.g., `ffmpeg -i input.mov -c:v libx264 output.mp4`).")
                    logger.error(f"Could not open video file: {input_video_path}")
                    return

                # Get and log video properties
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
                # Ensure even dimensions (required by some codecs)
                frame_width = frame_width if frame_width % 2 == 0 else frame_width + 1
                frame_height = frame_height if frame_height % 2 == 0 else frame_height + 1
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if not available
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames_to_process = total_frames // frame_skip
                logger.info(f"Input video properties: resolution={frame_width}x{frame_height}, fps={fps}, total_frames={total_frames}, frames_to_process={frames_to_process}")

                # Create output folder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                subfolder = output_subfolder.strip() or timestamp
                output_dir = BASE_DIR / "output_videos" / subfolder
                os.makedirs(output_dir, exist_ok=True)
                output_video_path = str(output_dir / f"processed_video_{timestamp}.mp4")
                st.info(f"Output video will be saved to: {output_video_path}")
                logger.info(f"Output video will be saved to: {output_video_path}")

                # Try codecs: mp4v first, then avc1
                codecs = ["mp4v", "avc1"]
                out = None
                for codec in codecs:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    logger.info(f"Attempting codec: {codec}")
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                    if out.isOpened():
                        logger.info(f"Video writer initialized with codec: {codec}")
                        break
                    logger.warning(f"Failed to initialize video writer with codec: {codec}")
                    out.release()
                    out = None

                if out is None:
                    st.error("Error: Could not initialize video writer with any codec (mp4v, avc1). Ensure FFmpeg is installed (`sudo apt-get install ffmpeg` on Linux, or add FFmpeg to PATH on Windows) and OpenCV is built with FFmpeg support. Check OpenCV build info with `cv2.getBuildInformation()` and look for 'FFMPEG: YES'. Reinstall OpenCV if needed: `pip install opencv-python-headless`.")
                    logger.error("Failed to initialize video writer with all codecs.")
                    cap.release()
                    return

                # Process video
                st.subheader("Processing Video")
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_count = 0
                processed_count = 0
                with st.spinner(f"Processing {frames_to_process} frames..."):
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_count % frame_skip == 0:
                            # Resize frame if needed
                            if resize_factor < 1.0:
                                frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                            results = run_inference(model, frame)
                            if results:
                                img_annotated = draw_boxes_on_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), results, class_map)
                                frame_out = cv2.cvtColor(np.array(img_annotated), cv2.COLOR_RGB2BGR)
                                logger.info(f"Frame {frame_count}: Inference successful, writing annotated frame")
                                out.write(frame_out)
                            else:
                                logger.warning(f"Frame {frame_count}: Inference failed, writing original frame")
                                out.write(frame)
                            processed_count += 1
                            # Update progress
                            progress = min(processed_count / frames_to_process, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"Processed {processed_count}/{frames_to_process} frames")
                        else:
                            # Write original frame for skipped frames
                            if resize_factor < 1.0:
                                frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                            out.write(frame)
                        frame_count += 1

                cap.release()
                out.release()
                logger.info(f"Output video saved to: {output_video_path}")

                # Verify output video
                output_cap = cv2.VideoCapture(output_video_path)
                if not output_cap.isOpened() or output_cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
                    st.error(f"Error: Output video is corrupt or empty. Check if FFmpeg is installed and supports the selected codec. Try playing the video with VLC after downloading. Output path: {output_video_path}")
                    logger.error(f"Output video is corrupt or empty: {output_video_path}")
                    output_cap.release()
                    return
                output_cap.release()

                # Provide download option (no display)
                st.subheader("Download Processed Video")
                if os.path.exists(output_video_path):
                    with open(output_video_path, "rb") as video_file:
                        output_video_bytes = video_file.read()
                    st.download_button(
                        label="Download Processed Video",
                        data=output_video_bytes,
                        file_name=f"processed_video_{timestamp}.mp4",
                        mime="video/mp4"
                    )
                    st.success(f"Video processing complete! Saved to: {output_video_path}. Download the video and open it in a media player like VLC.")
                    logger.info("Download button provided for processed video")
                else:
                    st.error(f"Error: Output video file was not created. Check logs for details. Output path: {output_video_path}")
                    logger.error(f"Output video file not found: {output_video_path}")

            except Exception as e:
                st.error(f"Error processing video: {str(e)}. Ensure FFmpeg is installed, the video format is supported (convert using `ffmpeg -i input.mov -c:v libx264 output.mp4` if needed), and sufficient disk space is available.")
                logger.error(f"Error processing video: {str(e)}")
            finally:
                # Clean up temporary input file
                if input_video_path and os.path.exists(input_video_path):
                    try:
                        os.unlink(input_video_path)
                        logger.info(f"Deleted temporary input file: {input_video_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete input file {input_video_path}: {str(e)}")

if __name__ == "__main__":
    main()