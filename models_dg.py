from imports import *
from utils import GeneralModel


class ExampleModel(GeneralModel):
    """
    First simple model copied from the lab.

    Parameters
    ----------
    name : string
        The name of the model
    input_shape : tuple
        The shape of the input data
    output_shape : int
        The number of classes
    seed : int, optional
        To guarantee reproducibility, by default seed
    """

    def __init__(
        self,
        name,
        input_shape,
        output_shape,
        epochs=200,
        batch_size=128,
        patience=5,
        seed=SEED,
    ):
        super().__init__()
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.model = None
        self.history = None

    def build(self):
        """
        Definition and building of the model
        """
        tf.random.set_seed(self.seed)

        # Build the neural network layer by layer
        input_layer = tfkl.Input(shape=self.input_shape, name="Input")

        x = tfkl.Conv2D(
            filters=32, kernel_size=3, padding="same", activation="relu", name="conv00"
        )(input_layer)
        x = tfkl.Conv2D(
            filters=32, kernel_size=3, padding="same", activation="relu", name="conv01"
        )(x)
        x = tfkl.MaxPooling2D(name="mp0")(x)

        x = tfkl.Conv2D(
            filters=64, kernel_size=3, padding="same", activation="relu", name="conv10"
        )(x)
        x = tfkl.Conv2D(
            filters=64, kernel_size=3, padding="same", activation="relu", name="conv11"
        )(x)
        x = tfkl.MaxPooling2D(name="mp1")(x)

        x = tfkl.Conv2D(
            filters=128, kernel_size=3, padding="same", activation="relu", name="conv20"
        )(x)
        x = tfkl.Conv2D(
            filters=128, kernel_size=3, padding="same", activation="relu", name="conv21"
        )(x)
        x = tfkl.MaxPooling2D(name="mp2")(x)

        x = tfkl.Conv2D(
            filters=256, kernel_size=3, padding="same", activation="relu", name="conv30"
        )(x)
        x = tfkl.Conv2D(
            filters=256, kernel_size=3, padding="same", activation="relu", name="conv31"
        )(x)
        x = tfkl.GlobalAveragePooling2D(name="gap")(x)

        output_layer = tfkl.Dense(
            units=self.output_shape, activation="softmax", name="Output"
        )(x)

        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name="Convnet")

    def compile(self):
        """
        Compile the model
        """
        self.model.compile(
            loss=tfk.losses.CategoricalCrossentropy(),
            optimizer=tfk.optimizers.Adam(weight_decay=5e-4),
            metrics=["accuracy"],
        )

    def train(self, X_train, y_train, X_val, y_val):
        """
        Performs model training
        """
        # Define early stopping callback
        early_stopping = tfk.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=self.patience,
            mode="max",
            restore_best_weights=True,
        )

        # Train the model and save its history
        self.history = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
        ).history
