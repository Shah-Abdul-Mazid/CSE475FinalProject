import gradio as gr
import cv2
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import os
import random
import tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# --- Grad-CAM dependencies ---
try:
    from yolo_cam.eigen_cam import EigenCAM
    from yolo_cam.utils.image import show_cam_on_image
except ImportError:
    print("Warning: yolo_cam transformation modules not found. Grad-CAM will be disabled.")
    EigenCAM = None

# --- Configuration & Paths ---
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Model Paths
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

# CSV Paths for Metrics
CSV_PATHS = {
    "YOLO10_with_SGD": BASE_DIR / "yolo_training" / "yolov10_SGD" / "overall_metrics.csv",
    "YOLO10_with_AdamW": BASE_DIR / "yolo_training" / "yolov10_AdamW" / "overall_metrics.csv",
    "YOLO10_with_Adamax": BASE_DIR / "yolo_training" / "yolov10_Adamax" / "overall_metrics.csv",
    "YOLO10_with_Adam": BASE_DIR / "yolo_training" / "yolov10_Adam" / "overall_metrics.csv",
    "YOLO12_with_SGD": BASE_DIR / "yolo_training" / "yolo12_SGD" / "overall_metrics.csv",
    "YOLO12_with_AdamW": BASE_DIR / "yolo_training" / "yolo12_AdamW" / "overall_metrics.csv",
    "YOLO12_with_Adamax": BASE_DIR / "yolo_training" / "yolo12_Adamax" / "overall_metrics.csv",
    "YOLO12_with_Adam": BASE_DIR / "yolo_training" / "yolo12_Adam" / "overall_metrics.csv",
}

# Dataset Path
ROOT_DATASET_PATH = BASE_DIR / "Raw Image" / "Raw Images"

# Class Mapping
CLASS_MAP = {0: 'Bike', 1: 'Bus', 2: 'Car', 3: 'Cng', 4: 'Cycle', 5: 'Mini-Truck', 6: 'People', 7: 'Rickshaw', 8: 'Truck'}

# --- Core Logic Functions ---

def get_model(name):
    path = MODEL_PATHS.get(name)
    if path and path.exists():
        return YOLO(path)
    return None

def predict_image(img, model_name, do_gradcam):
    if img is None: return None, None
    model = get_model(model_name)
    if not model: return None, "Model file not found. Please check paths."
    
    # 1. Detection
    results = model(img)
    annotated_img = results[0].plot() 
    
    # 2. Grad-CAM
    gradcam_res = None
    if do_gradcam and EigenCAM:
        try:
            target_layers = [model.model.model[-2]]
            cam = EigenCAM(model, target_layers, task='od')
            img_resized = np.array(img.resize((640, 640)))
            img_float = np.float32(img_resized) / 255.0
            grayscale_cam = cam(img_resized)[0, :, :]
            gradcam_res = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        except Exception as e:
            print(f"Grad-CAM Error: {e}")
            
    return annotated_img, gradcam_res

def process_video(video_path, model_name):
    if not video_path: return None
    model = get_model(model_name)
    if not model: return None
    
    # Process and save results
    results = model.predict(video_path, save=True, project="gradio_output", exist_ok=True)
    saved_path = os.path.join(results[0].save_dir, os.path.basename(video_path))
    return saved_path

def get_random_dataset_images():
    if not ROOT_DATASET_PATH.exists(): return []
    all_imgs = []
    for root, _, files in os.walk(ROOT_DATASET_PATH):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_imgs.append(os.path.join(root, f))
    
    if not all_imgs: return []
    samples = random.sample(all_imgs, min(8, len(all_imgs)))
    return samples

def load_metrics(model_name):
    csv_path = CSV_PATHS.get(model_name)
    if not csv_path or not csv_path.exists():
        return None, "Performance data (CSV) not found for this optimizer variant."
    
    df = pd.read_csv(csv_path)
    # Filter for plotting
    df_plot = df[df['Class Name'] != 'Overall Metrics']
    fig = px.bar(df_plot, x='Class Name', y='AP50', title=f"mAP@0.5 Comparison for {model_name}", 
                 color='Class Name', template="plotly_dark")
    return df, fig

# --- Gradio UI Layout ---

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"), title="Traffic AI Dashboard") as demo:
    gr.Markdown("# 🚗 Bangladeshi Traffic Flow Analysis Dashboard (Gradio Performance Edition)")
    gr.Markdown("Real-time vehicle detection and model interpretability (Grad-CAM) for researchers.")

    with gr.Tabs():
        # TAB 1: Image Detection
        with gr.TabItem("🖼️ Image Detection"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Input Image")
                    model_sel = gr.Dropdown(list(MODEL_PATHS.keys()), value="YOLO12_with_AdamW", label="Model Variant")
                    check_gc = gr.Checkbox(label="Show Grad-CAM Heatmap", value=True)
                    btn_run = gr.Button("Run Inference", variant="primary")
                with gr.Column():
                    out_detection = gr.Image(label="Annotated Detections")
                    out_gradcam = gr.Image(label="Grad-CAM Visualization")
            
            btn_run.click(predict_image, inputs=[img_input, model_sel, check_gc], outputs=[out_detection, out_gradcam])

        # TAB 2: Video Flow
        with gr.TabItem("🎥 Video Analytics"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Input MP4/AVI Video")
                    v_model_sel = gr.Dropdown(list(MODEL_PATHS.keys()), value="YOLO12_with_AdamW", label="Model Selection")
                    btn_v_run = gr.Button("Process Full Video", variant="primary")
                with gr.Column():
                    out_video = gr.Video(label="Generated Output")
            
            btn_v_run.click(process_video, inputs=[vid_input, v_model_sel], outputs=out_video)

        # TAB 3: Scientific Validation
        with gr.TabItem("📊 Performance Metrics"):
            with gr.Row():
                with gr.Column(scale=1):
                    m_sel = gr.Dropdown(list(MODEL_PATHS.keys()), label="Choose Model Variant")
                    m_btn = gr.Button("Load CSV Metrics")
                with gr.Column(scale=2):
                    metric_table = gr.Dataframe(label="Raw Evaluation Data")
            
            metric_plot = gr.Plot(label="Accuracy Visualization (mAP@0.5)")
            m_btn.click(load_metrics, inputs=m_sel, outputs=[metric_table, metric_plot])

        # TAB 4: Dataset Explorer
        with gr.TabItem("📁 Dataset Preview"):
            gr.Markdown("Samples from the specialized Bangladeshi Traffic dataset.")
            gallery = gr.Gallery(label="Traffic Scenarios", columns=4, height="600px")
            prev_btn = gr.Button("Refresh Samples")
            
            prev_btn.click(get_random_dataset_images, outputs=gallery)

    gr.Markdown("---")
    gr.Markdown("© 2024 Shah Abdul Mazid. Built for high-speed AI evaluation.")

if __name__ == "__main__":
    demo.launch(share=True)
