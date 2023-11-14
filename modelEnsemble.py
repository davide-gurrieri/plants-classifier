import os
import tensorflow as tf
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


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

        ensemble_labels = stats.mode(labels, axis=-1)[0]
        ensemble_labels = np.squeeze(ensemble_labels)

        # Debugging: Print shape of ensemble_labels before returning
        print(ensemble_labels.shape)

        return ensemble_labels

    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


    def evaluate(self, x_eval, y_eval, predictions):

        predictions = predictions.astype(int)[:, np.newaxis]
        accuracy = accuracy_score(y_eval, predictions)
        precision = precision_score(y_eval, predictions)
        recall = recall_score(y_eval, predictions)
        cm = confusion_matrix(y_eval, predictions)
        print(
            f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nConfusion matrix:\n{cm}"
        )

