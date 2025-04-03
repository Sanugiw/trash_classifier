import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_generator, epochs=10)
model.save('models/custom_trash_model.h5')

# In train.py (after model training completes)
model.fit(train_generator, epochs=10)

# 4. CONVERT TO TFLITE (ADD THIS BLOCK IMMEDIATELY AFTER TRAINING)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization for size/speed
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # Compatibility
tflite_model = converter.convert()

# Save the converted model
with open("models/trash_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite and saved to models/trash_model.tflite")

# Optional: Save class indices for labels
import json
with open('models/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)