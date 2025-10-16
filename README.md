# Deepfake Detection System

This repository combines both components of the Deepfake Detection Project:

##  1. Web Application (`deepfake-detector-web`)
- Flask backend for deepfake detection
- Frontend (HTML/CSS/JS) for uploads
- Uses pre-trained CNN/ONNX models for inference

##  2. Model Training (`deepfake-detector-modeltraining`)
- PyTorch-based training pipeline
- Scripts for dataset preparation, training, and ONNX export

##  How to Run
### Web App
```bash
cd deepfake-detector-web
pip install -r requirements.txt
python server.py
