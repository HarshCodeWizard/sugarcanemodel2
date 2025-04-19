from flask import Flask, request, jsonify
import keras
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the Keras model
model = keras.models.load_model("model.keras")

# Sugarcane disease classes - update these labels based on your specific model
labels = ["Healthy", "Red Rot", "Rust", "Smut", "Yellow Leaf Disease"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file is included in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process the image
        image = Image.open(file.stream).convert("RGB")
        image = image.resize((224, 224))  # Resize to match model input (224x224 based on your model config)
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        input_data = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Run inference
        predictions = model.predict(input_data)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_label = labels[predicted_index]
        confidence = float(predictions[0][predicted_index])
        
        return jsonify({
            'class': predicted_label,
            'confidence': round(confidence * 100, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')