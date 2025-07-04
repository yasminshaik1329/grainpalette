from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('rice.h5')

classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

def predict_rice_type(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)], np.max(prediction)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    rice_type, confidence = predict_rice_type(filepath)
    return render_template('results.html', rice_type=rice_type, confidence=confidence, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
