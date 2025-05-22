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

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import scale_cam_image, show_cam_on_image

# Configure logging
logging.basicConfig(
    filename="app.log",
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

from pathlib import Path
root_dataset_path = Path("Raw Image") / "Raw Images"
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


if not BASE_DIR.exists():
    st.error(f"Base directory not found: {BASE_DIR}")
    logger.error(f"Base directory not found: {BASE_DIR}")
    st.stop()
logger.info(f"Base Directory: {BASE_DIR}")

# Dataset configuration
DATASET_CONFIG = {
    "root_dataset_path": root_dataset_path,  # Updated to use the new base directory
    "locations": ["Location 1 (Arambag)", "Location 2 (Shapla Chattar)", "Location 3 (Abul Hotel)", "Location4 (Bashabo)"]  # Replace with actual location names
}

# Model paths - updated to be relative to the new base directory
MODEL_PATHS = {
    "YOLO10_with_SGD": BASE_DIR.parent.parent / "yolo_training" / "yolov10_SGD" / "weights" / "best.pt",
    "YOLO10_with_AdamW": BASE_DIR.parent.parent / "yolo_training" / "yolov10_AdamW" / "weights" / "best.pt",
    "YOLO10_with_Adamax": BASE_DIR.parent.parent / "yolo_training" / "yolov10_Adamax" / "weights" / "best.pt",
    "YOLO10_with_Adam": BASE_DIR.parent.parent / "yolo_training" / "yolov10_Adam" / "weights" / "best.pt",
    "YOLO12_with_SGD": BASE_DIR.parent.parent / "yolo_training" / "yolo12_SGD" / "weights" / "best.pt",
    "YOLO12_with_AdamW": BASE_DIR.parent.parent / "yolo_training" / "yolo12_AdamW" / "weights" / "best.pt",
    "YOLO12_with_Adamax": BASE_DIR.parent.parent / "yolo_training" / "yolo12_Adamax" / "weights" / "best.pt",
    "YOLO12_with_Adam": BASE_DIR.parent.parent / "yolo_training" / "yolo12_Adam" / "weights" / "best.pt",
}

# CSV paths for metrics - updated to be relative to the new base directory
csv_paths = {
    "YOLO10_with_SGD": BASE_DIR.parent.parent / "yolo_training" / "yolov10_SGD" / "overall_metrics.csv",
    "YOLO10_with_AdamW": BASE_DIR.parent.parent / "yolo_training" / "yolov10_AdamW" / "overall_metrics.csv",
    "YOLO10_with_Adamax": BASE_DIR.parent.parent / "yolo_training" / "yolov10_Adamax" / "overall_metrics.csv",
    "YOLO10_with_Adam": BASE_DIR.parent.parent / "yolo_training" / "yolov10_Adam" / "overall_metrics.csv",
    "YOLO12_with_SGD": BASE_DIR.parent.parent / "yolo_training" / "yolo12_SGD" / "overall_metrics.csv",
    "YOLO12_with_AdamW": BASE_DIR.parent.parent / "yolo_training" / "yolo12_AdamW" / "overall_metrics.csv",
    "YOLO12_with_Adamax": BASE_DIR.parent.parent / "yolo_training" / "yolo12_Adamax" / "overall_metrics.csv",
    "YOLO12_with_Adam": BASE_DIR.parent.parent / "yolo_training" / "yolo12_Adam" / "overall_metrics.csv"
}

# Image paths for evaluation plots - updated to be relative to the new base directory
IMAGE_PATHS_MAP = {
    "YOLO10_with_SGD": {
        "Normalized Confusion Matrix": BASE_DIR.parent.parent / "yolo_training" / "yolov10_SGD" / "confusion_matrix_normalized.png",
        "F1 Curve": BASE_DIR.parent.parent / "yolo_training" / "yolov10_SGD" / "F1_curve.png",
        "Precision Curve": BASE_DIR.parent.parent / "yolo_training" / "yolov10_SGD" / "P_curve.png",
        "Precision-Recall Curve": BASE_DIR.parent.parent / "yolo_training" / "yolov10_SGD" / "PR_curve.png",
        "Recall Curve": BASE_DIR.parent.parent / "yolo_training" / "yolov10_SGD" / "R_curve.png",
        "Results": BASE_DIR.parent.parent / "yolo_training" / "yolov10_SGD" / "results.png"
    },
    # Add similar entries for other models (omitted for brevity)
    # Ensure all models have corresponding entries as in the original code
}

# [Rest of the functions remain the same...]

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Machine Learning-MRAR",
        page_icon=str(BASE_DIR.parent.parent / "assets" / "artificial-intelligence.png"),  # Updated path
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
        st.markdown("""
        ## Project Overview
        This project uses **YOLOv10**, **YOLOv12**, and **Faster-RCNN_resnet50_fpn** models to detect and classify vehicles in **Bangladeshi traffic flow data**. It aims to enhance traffic monitoring, urban planning, and road safety in Bangladesh.

        ---

        ## Objectives
        - **Vehicle Detection & Classification**: Identify vehicle types (e.g., cars, buses, trucks, rickshaws).
        - **Traffic Pattern Analysis**: Provide insights into traffic density and vehicle distribution.
        - **Model Interpretability**: Use Grad-CAM to visualize model decision-making.
        - **User-Friendly Interface**: Support image uploads, real-time inference, and visualizations.

        ---

        ## Bangladeshi Traffic Flow Dataset
        - **Diverse Scenarios**: Urban intersections, rural roads, highways under varied conditions.
        - **Vehicle Classes**: Cars, buses, trucks, motorcycles, rickshaws, etc.
        - **Challenges**: Overcrowded scenes, occlusions, low visibility.

        ---

        ## Models
        - **YOLOv10**: Fast, lightweight, suitable for real-time detection.
        - **YOLOv12**: Advanced feature extraction for complex scenes.
        - **Faster-RCNN_resnet50_fpn**: High accuracy for challenging scenarios.

        ---

        ## Key Features
        - **Grad-CAM Visualization**: Highlights regions influencing predictions.
        - **Real-Time Inference**: Detects objects in webcam streams or uploaded videos.
        - **Interactive UI**: Model selection, image uploads, and performance metrics.

        ---

        ## Get Started
        Use the sidebar to:
        - **Model**: Upload images for inference and Grad-CAM.
        - **Dataset**: Preview dataset images.
        - **Results**: View model performance metrics.
        - **Real-time Detection**: Run live detection via webcam.
        - **Upload Video**: Process uploaded videos.

        ---

        ## Acknowledgments
        Thanks to the YOLO and Faster-RCNN communities, dataset contributors, and open-source ecosystem.

        ## Future Work
        - Optimize models for better accuracy and speed.
        - Integrate with traffic systems for real-time monitoring.
        - Add features like vehicle counting or speed estimation.
        """)
        if os.path.exists(BASE_DIR.parent.parent / "assets" / "poster.jpg"):
            st.image(BASE_DIR.parent.parent / "assets" / "poster.jpg", caption="Traffic in Dhaka, Bangladesh", use_container_width=True)
        st.markdown("---")
        if os.path.exists(BASE_DIR.parent.parent / "assets" / "developer.jpg"):
            st.markdown("""
                <div style='text-align: center; margin-top: 30px;'>
                    <img src='data:image/png;base64,{developer_img_base64}' width='300'><br>
                    <p style='margin-top: 10px; font-size: 16px;'>
                    <strong>Developed By</strong>:<br>
                    <em>Shah Abdul Mazid, Mahia Meherun Safa, Abir Sarwar, Raida Sultana</em>
                    </p>
                </div>
                """.format(
                    developer_img_base64=base64.b64encode(open(BASE_DIR.parent.parent / "assets" / "developer.jpg", "rb").read()).decode()
                ), unsafe_allow_html=True)

    elif selected == "Dataset":
        st.subheader("Dataset Preview")
        st.write("Preview random images from the Bangladeshi Traffic Flow Dataset.")
        num_images = st.number_input("Number of images to preview:", min_value=1, max_value=100, value=5)
        images_per_row = st.number_input("Images per row:", min_value=1, max_value=10, value=5)

        # Updated dataset path construction
        root_dataset_path = root_dataset_path
        
        @st.cache_data
        def get_image_paths(root_path):
            """Safely get all image paths from directory and subdirectories"""
            image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
            image_paths = []
            for dirpath, _, filenames in os.walk(root_path):
                for filename in filenames:
                    if filename.lower().endswith(image_extensions):
                        image_paths.append(Path(dirpath) / filename)
            return image_paths

        if root_dataset_path.exists():
            all_image_paths = get_image_paths(root_dataset_path)
            if all_image_paths:
                # Ensure we don't request more images than available
                num_samples = min(num_images, len(all_image_paths))
                samples = random.sample(all_image_paths, num_samples)
                
                # Display images in a grid
                for i in range(0, len(samples), images_per_row):
                    cols = st.columns(images_per_row)
                    for j, img_path in enumerate(samples[i:i+images_per_row]):
                        try:
                            with Image.open(img_path) as img:
                                cols[j].image(
                                    img,
                                    caption=img_path.name,
                                    use_container_width=True
                                )
                        except Exception as e:
                            cols[j].warning(f"Failed to load {img_path.name}: {str(e)}")
            else:
                st.warning(f"No images found in: {root_dataset_path}")
        else:
            st.error(f"Dataset directory not found at: {root_dataset_path}")

    elif selected == "Model":
        st.subheader("Run Inference on Uploaded Image")
        st.write("Upload a JPG, PNG, or JPEG image for object detection and Grad-CAM visualization.")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        model_choice = st.selectbox("Select YOLO Model", ["select a model"] + list(MODEL_PATHS.keys()))
        folder_name = st.text_input("Output folder name (optional, defaults to timestamp)", "")

        if uploaded_file and model_choice != "select a model":
            try:
                image = Image.open(uploaded_file).convert("RGB")
                model_path = MODEL_PATHS.get(model_choice)
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

                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            image.save(tmp_file.name)
                            temp_img_path = tmp_file.name

                        try:
                            gradcam_path = grad_cam_and_save(
                                model_path=model_path,
                                img_path=temp_img_path,
                                save_dir=save_dir,
                                use_multi_layer=True,
                                file_prefix=file_prefix
                            )
                            st.write(f"Grad-CAM output saved: {gradcam_path}")

                            def resize_image(image, target_width):
                                aspect_ratio = image.height / image.width
                                target_height = int(target_width * aspect_ratio)
                                return image.resize((target_width, target_height))

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
                        finally:
                            if os.path.exists(temp_img_path):
                                os.unlink(temp_img_path)
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
        metrics_cols = ['Precision', 'Recall', 'F1-Score', 'mAP@0.5', 'mAP@0.5:0.95']
        data_yaml = BASE_DIR / "Bd_Traffic_Dataset_v6" / "Bangladeshi Traffic Flow Dataset" / "Bangladeshi Traffic Flow Dataset" / "data.yaml"
        if not data_yaml.exists():
            st.error(f"Data YAML file not found: {data_yaml}")
            logger.error(f"Data YAML file not found: {data_yaml}")
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
                logger.info(f"No plots available for {model_name}")

    elif selected == "Real-time Detection":
        st.subheader("Real-time Object Detection")
        st.write("Perform object detection using your webcam. Click 'Stop Inference' to end the session.")
        model_choice_rt = st.selectbox("Select YOLO Model for Real-time Detection", ["select a model"] + list(MODEL_PATHS.keys()))
        video_source = st.number_input("Video Source Index", min_value=0, value=0)
        frame_size = st.slider("Frame Width", min_value=320, max_value=1280, value=640, step=32)

        if model_choice_rt != "select a model":
            model_path = MODEL_PATHS.get(model_choice_rt)
            model = get_model(model_path)
            if model:
                st.info("Starting webcam inference. Click 'Stop Inference' to stop.")
                real_time_inference(model, get_device(), video_source, frame_size)
            else:
                st.error("Model could not be loaded.")

    elif selected == "Upload Video":
        st.subheader("Upload a Video for Inference")
        st.write("Upload an MP4, AVI, MOV, or WEBM video for object detection.")
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "webm"])
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
                    logger.error(f"Failed to open video file: {input_video_path}")
                    return

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                codec = get_available_codec()
                if not codec:
                    st.error("No supported video codecs found.")
                    logger.error("No supported video codecs found")
                    return
                fourcc = cv2.VideoWriter_fourcc(*codec)
                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
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
                            logger.info(f"Processed {frame_count}/{total_frames} frames")

                cap.release()
                out.release()
                st.success("Video processing complete!")
                st.video(output_video_path)
                logger.info("Video processing completed successfully")

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                logger.error(f"Error processing video: {str(e)}")
            finally:
                if input_video_path and os.path.exists(input_video_path):
                    os.unlink(input_video_path)
                if output_video_path and os.path.exists(output_video_path):
                    os.unlink(output_video_path)
                logger.info("Cleaned up temporary video files")

if __name__ == "__main__":
    main()