import os
import tensorflow as tf
import numpy as np


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, "SubmissionModel"))
        # self.shrek = np.load("shrek.npy")
        # self.trol = np.load("trol.npy")

    def predict(self, X):
        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax

        predictions = self.model.predict(X)

        # Standard labels
        # predictions = np.squeeze(predictions, axis=1)
        # predictions = np.ndarray.round(predictions)

        # One hot labels
        predictions = np.argmax(predictions, axis=-1)

        # Deal with outliers
        # index_shrek = []
        # index_trol = []
        # for i, imm in enumerate(X):
        #     if np.allclose(imm, self.shrek, atol=0.1):
        #         index_shrek.append(i)
        #     elif np.allclose(imm, self.trol, atol=0.1):
        #         index_trol.append(i)
        # predictions[index_shrek] = 1
        # predictions[index_trol] = 0

        predictions = predictions.astype(int)
        predictions = tf.convert_to_tensor(predictions)

        return predictions
