from flask import Flask, request, jsonify
import onnxruntime as ort # Use onnxruntime
import numpy as np
from PIL import Image
import io
import os # Import os to get PORT environment variable

app = Flask(__name__)

# --- ONNX Model Loading ---
try:
    # **IMPORTANT**: Replace "model.onnx" with the actual filename of your ONNX model
    onnx_model_path = "model.onnx" 
    
    print(f"Loading ONNX model from: {onnx_model_path}")
    # Create an inference session, explicitly using CPU Execution Provider
    sess_options = ort.SessionOptions()
    # Optional: Optimizations (can sometimes speed up inference)
    # sess_options.intra_op_num_threads = 1 
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
    
    # Get input and output names from the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Model Input Name: {input_name}")
    print(f"Model Output Name: {output_name}")
    
    # Get expected input shape (useful for validation, often includes batch size 'None')
    input_shape = session.get_inputs()[0].shape 
    print(f"Model Expected Input Shape: {input_shape}")
    # Extract expected height and width (assuming shape is like [None, H, W, C] or [None, C, H, W])
    # Adjust indices based on your model's specific input format (e.g., [1, 2] for BHWC or [2, 3] for BCHW)
    try:
        # Assuming BHWC format like Keras default [None, 224, 224, 3]
        expected_height = input_shape[1]
        expected_width = input_shape[2]
        print(f"Using Input Size: {expected_height}x{expected_width}")
    except (IndexError, TypeError):
        print("Could not automatically determine input size from model shape, defaulting to 224x224.")
        # ** Fallback/Default size if shape info is unusual **
        expected_height, expected_width = 224, 224 


except Exception as e:
    print(f"!!!!!!!!!!!!!!!!! ERROR LOADING ONNX MODEL !!!!!!!!!!!!!!!!!!")
    print(f"Error: {e}")
    print(f"Ensure '{onnx_model_path}' exists and is a valid ONNX file.")
    # Exit or raise if model loading fails, as the app can't function
    raise RuntimeError(f"Failed to load ONNX model: {e}") from e
# -------------------------


# Sugarcane disease classes - update these labels based on your specific model
labels = ["Healthy", "Red Rot", "Rust", "Smut", "Yellow Leaf Disease"] # Make sure this matches your ONNX model's output classes

@app.route('/predict', methods=['POST'])
def predict():
    if not session: # Check if model loaded successfully
         return jsonify({'error': 'Model not loaded'}), 500
         
    try:
        # Check if an image file is included in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process the image
        print("Processing image...")
        image = Image.open(file.stream).convert("RGB")
        
        # Resize to the dimensions the ONNX model expects
        image = image.resize((expected_width, expected_height)) 
        
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension and ensure dtype is float32 (ONNX often requires this)
        input_data = np.expand_dims(image_array, axis=0).astype(np.float32) 
        
        print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}") # Should be (1, H, W, 3) and float32
        
        # Run inference using ONNX Runtime
        print("Running inference...")
        onnx_outputs = session.run([output_name], {input_name: input_data})
        predictions = onnx_outputs[0] # Output is usually a list of arrays
        
        print(f"Raw predictions shape: {predictions.shape}") # Should be (1, num_classes)
        
        # Process predictions
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])
        
        if predicted_index < len(labels):
             predicted_label = labels[predicted_index]
        else:
             predicted_label = "Unknown Class" # Handle index out of bounds
             print(f"Warning: Predicted index {predicted_index} is out of bounds for labels list (length {len(labels)})")

        print(f"Prediction: {predicted_label}, Confidence: {confidence:.4f}")
        
        return jsonify({
            'class': predicted_label,
            'confidence': round(confidence * 100, 2) # Return confidence as percentage
        })
    
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!! ERROR DURING PREDICTION !!!!!!!!!!!!!!!!!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to logs
        return jsonify({'error': f"An internal error occurred: {e}"}), 500

@app.route('/', methods=['GET'])
def health_check():
    # Simple endpoint to check if the server is running
    return jsonify({"status": "ok", "message": "API is running"}), 200

if __name__ == '__main__':
    # Get port from environment variable or default to 10000 for Render/local dev
    port = int(os.environ.get('PORT', 10000)) 
    # Set debug=False for production on Render
    # Host '0.0.0.0' makes it accessible externally
    print(f"Starting Flask server on host 0.0.0.0 port {port}")
    app.run(debug=False, host='0.0.0.0', port=port) 
