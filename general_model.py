from imports import *


class GeneralModel:
    name = ""
    seed = SEED
    model = tfk.Model()
    history = {}
    history_val = {}
    cv_n_fold = 0
    cv_histories = []
    cv_scores = []
    cv_best_epochs = []
    cv_avg_epochs = -1
    augmentation = tf.keras.Sequential(
        [
            tfkl.RandomFlip(mode="horizontal"),
            tfkl.RandomFlip(mode="vertical"),
            tfkl.RandomRotation(factor=0.25),
            # tfkl.RandomContrast(factor=0.8),
        ],
        name="preprocessing",
    )

    def __init__(self, build_kwargs={}, compile_kwargs={}, fit_kwargs={}):
        self.build_kwargs = build_kwargs
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs

    def build(self):
        pass

    def compile(self):
        """
        Compile the model
        """
        self.model.compile(**self.compile_kwargs)

    def train(
        self,
        x_train,
        y_train,
        one_hot=True,
        balanced=False,
        loss_weights=(1, 1),
    ):
        """
        Train the model
        """
        if balanced:
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(y_train[:, 0]), y=y_train[:, 0]
            )
            self.fit_kwargs["class_weight"] = {
                0: loss_weights[0] * class_weights[0],
                1: loss_weights[1] * class_weights[1],
            }

        if one_hot:
            y_train = tfk.utils.to_categorical(y_train)

        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            **self.fit_kwargs,
        ).history

    def train_val(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        one_hot=True,
        balanced=False,
        loss_weights=(1, 1),
    ):
        """
        Train the model
        """
        if balanced:
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(y_train[:, 0]), y=y_train[:, 0]
            )
            self.fit_kwargs["class_weight"] = {
                0: loss_weights[0] * class_weights[0],
                1: loss_weights[1] * class_weights[1],
            }

        if one_hot:
            y_train = tfk.utils.to_categorical(y_train)
            y_val = tfk.utils.to_categorical(y_val)

        self.history_val = self.model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            **self.fit_kwargs,
        ).history

    def save_model(self):
        """
        Save the trained model in the models folder
        """
        self.model.save(f"saved_models/{self.name}")

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
        keys = list(self.history_val.keys())
        n_metrics = len(keys) // 2

        for i in range(n_metrics):
            plt.figure(figsize=figsize)
            if training:
                plt.plot(
                    self.history_val[keys[i]], label="Training " + keys[i], alpha=0.8
                )
            plt.plot(
                self.history_val[keys[i + n_metrics]],
                label="Validation " + keys[i],
                alpha=0.8,
            )
            plt.title(keys[i])
            plt.legend()
            plt.grid(alpha=0.3)

        plt.show()

    def evaluate(self, x_eval, y_eval):
        """
        Evaluate the model on the evaluation set.

        Parameters
        ----------
        X_eval : numpy.ndarray
            Evaluation input data
        y_eval : numpy.ndarray
            Evaluation target data
        """
        predictions = self.model.predict(x_eval, verbose=0)
        if self.build_kwargs["output_shape"] == 2:
            # predictions is a n x 2 matrix
            predictions = np.argmax(predictions, axis=-1)
        else:
            # predictions is a probability vector
            predictions = np.ndarray.round(predictions).astype(int)
        accuracy = accuracy_score(y_eval, predictions)
        precision = precision_score(y_eval, predictions)
        recall = recall_score(y_eval, predictions)
        cm = confusion_matrix(y_eval, predictions)
        print(
            f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nConfusion matrix:\n{cm}"
        )
        return accuracy

    def train_cv(
        self,
        x_train_val,
        y_train_val,
        model_constructor,
        num_folds=10,
        one_hot=False,
        balanced=False,
        loss_weights=(1, 1),
    ):
        if balanced:
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(y_train_val[:, 0]), y=y_train_val[:, 0]
            )
            self.fit_kwargs["class_weight"] = {
                0: loss_weights[0] * class_weights[0],
                1: loss_weights[1] * class_weights[1],
            }

        self.cv_n_fold = num_folds
        # Create a cross-validation object
        kfold = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=self.seed
        )

        build_kwargs = dict(self.build_kwargs)
        compile_kwargs = dict(self.compile_kwargs)
        fit_kwargs = dict(self.fit_kwargs)

        # Loop through each fold
        for fold_idx, (train_idx, valid_idx) in enumerate(
            kfold.split(x_train_val, y_train_val)
        ):
            print(f"Starting training on fold num: {fold_idx + 1}")

            # Build a new model for each fold

            model_obj = model_constructor(
                self.name + "_fold_" + f"{fold_idx+1}",
                build_kwargs.copy(),
                compile_kwargs.copy(),
                fit_kwargs.copy(),
            )
            model_obj.build()
            model_obj.compile()
            model_obj.train_val(
                x_train_val[train_idx],
                y_train_val[train_idx],
                x_train_val[valid_idx],
                y_train_val[valid_idx],
                one_hot=one_hot,
                balanced=False,
            )

            # Evaluate the model on the validation data for this fold
            # Returns the loss value & metrics values for the model in test mode.
            score = model_obj.evaluate(
                x_train_val[valid_idx],
                y_train_val[valid_idx],
            )
            self.cv_scores.append(score)

            # Calculate the best epoch for early stopping
            best_epoch = (
                len(model_obj.history_val["loss"])
                - model_obj.fit_kwargs["callbacks"][0].patience
            )

            self.cv_best_epochs.append(best_epoch)

            # Store the training history for this fold
            self.cv_histories.append(model_obj.history_val)

        # Print mean and standard deviation of Accuracy scores
        print("Score statistics:")
        print(
            f"Mean: {np.mean(self.cv_scores).round(4)}\nStd:  {np.std(self.cv_scores).round(4)}"
        )

        # Calculate the average best epoch (la patience viene sottratta)
        self.cv_avg_epochs = int(np.mean(self.cv_best_epochs))
        print(f"Best average number of epochs: {self.cv_avg_epochs}")

        # train on the entire dataset
        print("Training on the entire dataset...")
        self.build()
        self.compile()
        self.train(x_train_val, y_train_val, one_hot=one_hot, balanced=False)

    def plot_cv_histories(self):
        # Define a list of colors for plotting
        colors = sns.color_palette("husl", self.cv_n_fold)

        # Create a figure for MSE visualization
        plt.figure(figsize=(15, 6))

        # Plot Accuracy for each fold
        patience = self.fit_kwargs["callbacks"][0].patience
        for fold_idx in range(self.cv_n_fold):
            plt.plot(
                self.cv_histories[fold_idx]["val_accuracy"][:-patience],
                color=colors[fold_idx],
                label=f"Fold N°{fold_idx+1}",
            )
            plt.title("Accuracy")
            plt.legend(loc="upper left")
            plt.grid(alpha=0.3)

        # Show the plot
        plt.show()
