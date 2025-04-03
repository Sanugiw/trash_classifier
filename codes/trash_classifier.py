from PIL import Image
import tensorflow.lite as tflite
import numpy as np
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model
# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "mobilenet_v1_1.0_224_quant.tflite")

interpreter = tflite.Interpreter(model_path=model_path)  # Fixed path
interpreter.allocate_tensors()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def classify_image(image_path):
    img = Image.open(image_path).convert('RGB')  # Ensure RGB format
    img = img.resize((224, 224), Image.Resampling.LANCZOS)  # Better resampling
    
    # Center crop if needed (for better focus)
    width, height = img.size
    new_size = min(width, height)
    left = (width - new_size)/2
    top = (height - new_size)/2
    img = img.crop((left, top, left + new_size, top + new_size))
    
    input_data = np.expand_dims(img, axis=0).astype(np.uint8)
    
    # Get input details to verify type
    input_details = interpreter.get_input_details()
    print(f"Model expects: {input_details[0]['dtype']}")  # Should show uint8
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    
    predicted_class = labels[np.argmax(output_data[0])]
    confidence = np.max(output_data[0]) / 255.0  # Convert uint8 output to 0-1 range
    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    temp_path = "temp_upload.jpg"
    file.save(temp_path)
    class_name, confidence = classify_image(temp_path)
    os.remove(temp_path)
    
    return jsonify({
        "class": class_name,
        "confidence": f"{confidence:.2%}",
        "recyclable": class_name.lower() in ["plastic", "paper", "metal", "glass", "cardboard"]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)