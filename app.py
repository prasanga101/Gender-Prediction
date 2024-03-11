# app.py
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)

# Load the saved model
def load_custom_model():
    model_path = "my_models.h5"  # Path to your saved Keras model
    model = load_model(model_path)
    return model

model = load_custom_model()

# Preprocess the image
def preprocess_image(image_data):
    image_stream = io.BytesIO(base64.b64decode(image_data.split(',')[1]))
    img = Image.open(image_stream)
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1,) + img_array.shape)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_data = request.json['image_data']
        image_array = preprocess_image(image_data)
        prediction = model.predict(image_array)
        result = "Male" if prediction[0][0] > 0.5 else "Female"
        return jsonify({'result': result})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
