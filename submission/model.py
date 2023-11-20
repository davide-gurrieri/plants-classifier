import os
import tensorflow as tf
import numpy as np


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, "SubmissionModel"))

    def predict(self, X):

        predictions = self.model.predict(X)

        # Standard labels
        predictions = np.squeeze(predictions, axis=1)
        predictions = np.ndarray.round(predictions)

        # One hot labels
        # predictions = np.argmax(predictions, axis=-1)

        predictions = predictions.astype(int)
        predictions = tf.convert_to_tensor(predictions)

        return predictions
