from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model.h5")
classes = ['bird','cat','deer','dog','frog','horse']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No image uploaded")

    file = request.files['image']

    # ✅ FIX: Convert FileStorage to BytesIO
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).resize((32,32))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    result = classes[np.argmax(prediction)]

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)