import os
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

app = Flask(__name__)

# Load ML model once
model = MobileNetV2(weights="imagenet")

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="Please choose an image")

        # Open image
        image = Image.open(file)
        processed = preprocess_image(image)

        # Predict
        preds = model.predict(processed)
        decoded = decode_predictions(preds, top=3)[0]

        predictions = [
            {"label": label, "confidence": float(score)}
            for (_, label, score) in decoded
        ]

    return render_template("index.html", predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)
