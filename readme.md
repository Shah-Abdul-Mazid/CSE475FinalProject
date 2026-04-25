# 🚗 Bangladeshi Traffic Flow Object Detection System

This project is a deep learning-based solution for detecting and classifying vehicles in the unique traffic conditions of Bangladesh. It leverages advanced YOLO architectures and provides model interpretability through Grad-CAM visualizations.

## 🌟 Key Features
- **Multi-Model Support**: Compare performance across YOLOv10 and YOLOv12 with different optimizers (SGD, Adam, Adamax, AdamW).
- **Explainable AI (XAI)**: Integrated **Grad-CAM** heatmaps to visualize exactly where the model is looking to identify objects.
- **High-Performance Dashboard**: A Gradio-based web interface for instant image and video inference.
- **Scientific Validation**: Built-in metrics viewer for mAP@0.5, Precision, and Recall across all vehicle classes.
- **Real-Time Ready**: Optimized for low-latency detection in complex urban scenarios.

## 🛠️ Tech Stack
- **Frameworks**: PyTorch, Ultralytics (YOLOv10/v12)
- **Interpretability**: yolo-cam (EigenCAM)
- **Deployment**: Gradio, Streamlit
- **Data Processing**: Pandas, OpenCV, NumPy
- **Visualization**: Plotly, Matplotlib

## 📂 Project Structure
- `app_final.py`: The main Gradio application (Optimized).
- `app.py`: Streamlit dashboard version.
- `yolo_training/`: Contains weights (`best.pt`) and evaluation metrics for each model variant.
- `Raw Image/`: The specialized dataset of Bangladeshi traffic.
- `yolo_cam/`: Custom implementation for Grad-CAM support.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install gradio ultralytics torch torchvision opencv-python pandas plotly pillow
