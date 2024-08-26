# # app.py
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import json

# app = Flask(__name__)

# # Load the trained model
# model = load_model('my_trained_model.keras')

# # Load class labels from the JSON file
# with open('class_labels.json', 'r') as f:
#     class_labels = json.load(f)

# def preprocess_image(img):
#     # Resize the image to match the model's input shape
#     img = img.resize((150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array /= 255.0  # Rescale the image
#     return img_array

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Check if an image was uploaded
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     # Get the file from the request
#     file = request.files['file']

#     # Open the image file
#     img = image.load_img(file)

#     # Preprocess the image
#     img_array = preprocess_image(img)

#     # Make prediction
#     predictions = model.predict(img_array)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     predicted_class_label = class_labels[predicted_class_index]

#     # Return the predicted class as JSON
#     return jsonify({'predicted_class': predicted_class_label})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)





from tensorflow import keras
from flask import Flask, request, jsonify
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import json
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model (adjust path if necessary)
model = load_model('my_trained_model.h5')

# Load class labels from the JSON file
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # Convert file to a PIL image
            img = Image.open(io.BytesIO(file.read()))
            img = img.resize((150, 150))  # Update size according to your model input
            img_array = np.array(img) / 255.0  # Normalize image data
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make a prediction using the model
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]

            return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
