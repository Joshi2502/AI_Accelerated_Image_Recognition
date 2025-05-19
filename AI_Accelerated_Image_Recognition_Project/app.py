
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from model import load_model, classify_image

app = Flask(__name__)

# Load default model
model = load_model()

@app.route('/train', methods=['POST'])
def train_model():
    from train import train
    data = request.json
    epochs = data.get('epochs', 5)
    lr = data.get('learning_rate', 0.001)
    train(epochs=epochs, lr=lr)
    return jsonify({'message': 'Model trained successfully'}), 200

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    prediction = classify_image(img, model)
    return jsonify({'prediction': prediction}), 200

@app.route('/load_model', methods=['POST'])
def load_model_api():
    global model
    model_version = request.json.get('version', 'default')
    model = load_model(version=model_version)
    return jsonify({'message': f'Model {model_version} loaded successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
