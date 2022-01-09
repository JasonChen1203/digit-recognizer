from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
import cv2
import base64


# Initialize flask
app = Flask(__name__, template_folder='display')

# Load prebuilt model
model = keras.models.load_model('src/Model/final_model')

# Handle GET request
@app.route('/', methods=['GET'])
def display():
    return render_template('index.html')


# Handle POST request
@app.route('/', methods=['POST'])
def canvas():
    # Recieve data from user
    canvasdata = request.form['canvasimg']
    encoded_data = request.form['canvasimg'].split(',')[1]

    # Decode base64 image to python array
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    rgb_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Change image to grayscale and resize to (28, 28)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (28, 28))

    # Change img type to feed into model
    img = np.expand_dims(gray_image, axis=0)
    img = img.reshape((1, 28, 28, 1))
    img_norm = img.astype('float32')
    img_norm /= 255.0

    try:
        prediction = np.argmax(model.predict(img_norm))
        print(f"Prediction Result : {str(prediction)}")

        return render_template('index.html', response=str(prediction), canvasdata=canvasdata)
    except Exception as e:
        return render_template('index.html', response=str(e), canvasdata=canvasdata)