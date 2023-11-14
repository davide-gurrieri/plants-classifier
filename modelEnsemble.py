import os
import tensorflow as tf
import numpy as np


class modelEnsamble:
    def __init__(self, model_paths):
        self.models = [tf.keras.models.load_model(model_path) for model_path in model_paths]


    def predict(self, X):
        
        labels=[]
        
        for m in self.models:
            predictions = np.round(m.predict(X,axis=1))
            labels.append(predictions)

        
        labels = np.array(labels)
        labels = np.transpose(labels, (1, 0))
        ensemble_labels = mode(labels, axis=-1)[0]
        ensemble_labels = np.squeeze(ensemble_labels)


        return ensemble_labels
