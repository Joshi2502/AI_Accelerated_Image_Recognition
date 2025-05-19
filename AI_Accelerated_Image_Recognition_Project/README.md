
# AI-Accelerated Image Recognition System

## Project Overview
This project is an advanced image recognition system using ResNet50 and PyTorch, accessible via a Flask API.

## Features
- Model Training via API (/train)
- Real-time Image Classification (/predict)
- Model Loading and Versioning (/load_model)

## Setup Instructions
1. Build the Docker image:
   docker build -t ai-image-recognition .

2. Run the Docker container:
   docker run -p 5000:5000 ai-image-recognition

## API Usage
- /train (POST): Train a model (Specify epochs, learning rate).
- /predict (POST): Classify an image (Upload image file).
- /load_model (POST): Load a specific model version.

