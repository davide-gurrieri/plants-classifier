import os
import tensorflow as tf
import numpy as np
from scipy import stats


class modelEnsemble:
    def __init__(self, model_paths):
        self.models = [tf.keras.models.load_model(model_path) for model_path in model_paths]


    def predict(self, X):
        
        labels=[]
        
        for m in self.models:
            predictions = np.round(m.predict(X))
            predictions = np.squeeze(predictions)
            labels.append(predictions)

        # Debugging: Print shapes of predictions
        for predictions in labels:
            print(predictions.shape)

        labels = np.array(labels)
        labels = np.transpose(labels, (1, 0))

        # Debugging: Print shape of labels before mode
        print(labels.shape)

        ensemble_labels = mode(labels, axis=-1)[0]
        ensemble_labels = np.squeeze(ensemble_labels)

        # Debugging: Print shape of ensemble_labels before returning
        print(ensemble_labels.shape)

        return ensemble_labels
