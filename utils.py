"""
This module contains utilities for the ANN course.
"""

from imports import *


class GeneralModel:
    n_folds = None

    def __init__(self):
        self.name = None
        self.epochs = None
        self.batch_size = None
        self.patience = 0
        self.seed = SEED
        self.model = None
        self.history = {}
        # cv data
        self.histories = []
        self.scores = []
        self.best_epochs = []
        self.avg_epochs = None

    def build(self):
        """
        Definition and building of the model
        """
        raise NotImplementedError

    def compile(self):
        """
        Compile the model
        """
        raise NotImplementedError

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model
        """
        raise NotImplementedError

    def save_model(self):
        """
        Save the trained model in the models folder
        """
        self.model.save(f"models/{self.name}")

    def plot_history(self, training=True, figsize=(15, 2)):
        """
        Plot the loss and metrics for the training and validation sets with respect to the training epochs.

        Parameters
        ----------
        training : bool, optional
            show the training plots, by default True
        figsize : tuple, optional
            dimension of the plots, by default (15, 2)
        """
        keys = list(self.history.keys())
        n_metrics = len(keys) // 2

        for i in range(n_metrics):
            plt.figure(figsize=figsize)
            if training:
                plt.plot(self.history[keys[i]], label="Training " + keys[i], alpha=0.8)
            plt.plot(
                self.history[keys[i + n_metrics]],
                label="Validation " + keys[i],
                alpha=0.8,
            )
            plt.title(keys[i])
            plt.legend()
            plt.grid(alpha=0.3)

        plt.show()

    def evaluate(self, X_eval, y_eval):
        """
        Evaluate the model on the evaluation set.

        Parameters
        ----------
        X_eval : numpy.ndarray
            Evaluation input data
        y_eval : numpy.ndarray
            Evaluation target data
        """
        # Predict labels for the entire validation set
        predictions = self.model.predict(X_eval, verbose=0)
        accuracy = accuracy_score(
            np.argmax(y_eval, axis=-1), np.argmax(predictions, axis=-1)
        )

        # Validation accuracy
        print(f"Evaluated accuracy: {accuracy:.4f}")

    def train_cv(
        self,
        x_train_val,
        y_train_val,
        num_folds=10,
        stratified=True,
        shuffle=True,
    ):
        self.n_folds = num_folds
        # Create a cross-validation object
        if stratified:
            kfold = StratifiedKFold(
                n_splits=num_folds, shuffle=shuffle, random_state=self.seed
            )
        else:
            kfold = KFold(n_splits=num_folds, shuffle=shuffle, random_state=self.seed)

        # Loop through each fold
        for fold_idx, (train_idx, valid_idx) in enumerate(
            kfold.split(x_train_val, y_train_val)
        ):
            print(f"Starting training on fold num: {fold_idx + 1}")

            # Build a new model for each fold
            self.build()
            self.compile()
            self.train(
                x_train_val[train_idx],
                tfk.utils.to_categorical(y_train_val[train_idx]),
                x_train_val[valid_idx],
                tfk.utils.to_categorical(y_train_val[valid_idx]),
            )

            # Evaluate the model on the validation data for this fold
            # Returns the loss value & metrics values for the model in test mode.
            score = self.model.evaluate(
                x_train_val[valid_idx],
                tfk.utils.to_categorical(y_train_val[valid_idx]),
                verbose=0,
            )
            self.scores.append(
                score[1]
            )  # score[0] is the loss, score[1] is the first metric

            # Calculate the best epoch for early stopping
            best_epoch = len(self.history["loss"]) - self.patience
            self.best_epochs.append(best_epoch)

            # Store the training history for this fold
            self.histories.append(self.history)

        # Print mean and standard deviation of Accuracy scores
        print("Score statistics:")
        print(
            f"Mean: {np.mean(self.scores).round(4)}\nStd:  {np.std(self.scores).round(4)}"
        )

        # Calculate the average best epoch (la patience viene sottratta)
        self.avg_epochs = int(np.mean(self.best_epochs))
        print(f"Best average number of epochs: {self.avg_epochs}")

        # train on the entire dataset
        print("Training on the entire dataset...")
        self.build()
        self.compile()
        self.train(x_train_val, tfk.utils.to_categorical(y_train_val), None, None)

    def plot_cv_histories(self):
        # Define a list of colors for plotting
        colors = sns.color_palette("husl", self.n_folds)

        # Create a figure for MSE visualization
        plt.figure(figsize=(15, 6))

        # Plot MSE for each fold
        for fold_idx in range(self.n_folds):
            plt.plot(
                self.histories[fold_idx]["val_mse"][: -self.patience],
                color=colors[fold_idx],
                label=f"Fold N°{fold_idx+1}",
            )
            plt.title("Accuracy")
            plt.legend(loc="upper left")
            plt.grid(alpha=0.3)

        # Show the plot
        plt.show()


# TO DO
# save object e load object
