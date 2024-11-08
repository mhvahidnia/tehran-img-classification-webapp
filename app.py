from flask import Flask, request, jsonify
import numpy as np
import json
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load the class labels
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Debugging: Check the type and content of class_labels
print(type(class_labels))  # Debugging
print(class_labels)  # Debugging

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='my_trained_mobilenet_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Debugging: Print output details to understand its structure
print(output_details)  # Debugging

def preprocess_image(image_path, target_size):
    """Preprocess the image to be compatible with the model input."""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Save the file temporarily
    file_path = 'temp_image.jpg'
    file.save(file_path)

    # Preprocess the image
    img = preprocess_image(file_path, target_size=(224, 224))  # Adjust size based on your model

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get the output tensor, assuming there is only one output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)[0]

    # Access the class label using an integer index
    predicted_label = class_labels[predicted_class]

    return jsonify({'predicted_class': predicted_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
