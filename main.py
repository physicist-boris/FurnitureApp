from fastapi import FastAPI, UploadFile
import tensorflow as tf
import numpy as np
import os
import io
from PIL import Image, ImageOps
from uvicorn import run
from modules.classifier import run_classifier


app_path = os.path.dirname(__file__)
app = FastAPI(title='Image Classification API')

@app.post("/predict/furnitureimage")
def predict_api(file: UploadFile):
    contents = file.file.read()
    buffer_contents = io.BytesIO(contents)
    img_pil = Image.open(buffer_contents)
    img_pil = ImageOps.fit(img_pil, (256, 256), method= Image.NEAREST)
    img_array = tf.keras.utils.img_to_array(img_pil).reshape(-1, 256, 256, 3)
    model_path = os.path.join(app_path, 'models/ml_model')
    results = run_classifier(img_array, path_to_model = model_path)
    return {
        "model-prediction": results[0],
        "model-prediction-confidence-score": results[1]
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run("main:app", host="0.0.0.0", port=port, reload=True)