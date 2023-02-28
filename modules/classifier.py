import tensorflow as tf
import numpy as np


def run_classifier(image_array, path_to_model):
    class_predictions = ['Bed', 'Chair', 'Sofa']
    new_model = tf.keras.models.load_model(path_to_model)
    score = new_model.predict(image_array)
    class_prediction = class_predictions[np.argmax(score)]
    model_score = np.max(score) * 100
    return class_prediction, model_score