# Load custom labels from training
with open('datasets/train/class_indices.json', 'r') as f:
    class_indices = json.load(f)
labels = {v:k for k,v in class_indices.items()}

# Load custom model
interpreter = tflite.Interpreter(model_path="models/trash_model.tflite")