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

        preprocessing = tf.keras.Sequential([
            tfkl.RandomFlip(mode="horizontal"),
            tfkl.RandomFlip(mode="vertical"),
            tfkl.RandomRotation(factor=0.25)
            tfkl.RandomContrast(factor=0.8)
        ], name='preprocessing')

        # Build the neural network layer by layer
        input_layer = tfkl.Input(shape=self.input_shape, name="Input")
        
        preprocessing = preprocessing(input_layer)

        x = tfkl.Conv2D(
            filters=32, kernel_size=3, padding="same", activation="relu", name="conv00"
        )(preprocessing)
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
        validation_data = None if X_val or y_val is None else (X_val, y_val)
        self.history = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=[early_stopping],
        ).history


class XceptionModel(GeneralModel):
    """
    Transfer learning model based on Xception.
    """

    def __init__(
        self,
        name,
        input_shape,
        output_shape,
        epochs=20,
        batch_size=32,
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

        data_augmentation = tfk.Sequential(
            [
                tfkl.RandomFlip("horizontal"),
                tfkl.RandomRotation(0.1),
            ]
        )

        base_model = tfk.applications.Xception(
            weights="imagenet",  # Load weights pre-trained on ImageNet.
            input_shape=self.input_shape,
            include_top=False,
        )  # Do not include the ImageNet classifier at the top.

        # Create new model on top
        inputs = tfkl.Input(shape=self.input_shape, name="Input")
        x = data_augmentation(inputs)  # Apply random data augmentation

        # Pre-trained Xception weights requires that input be scaled
        # from (0, 255) to a range of (-1., +1.), the rescaling layer
        # outputs: `(inputs * scale) + offset`
        scale_layer = tfkl.Rescaling(scale=1 / 127.5, offset=-1)
        x = scale_layer(x)

        # The base model contains batchnorm layers. We want to keep them in inference mode
        # when we unfreeze the base model for fine-tuning, so we make sure that the
        # base_model is running in inference mode here.
        x = base_model(x, training=False)
        x = tfkl.GlobalAveragePooling2D()(x)
        x = tfkl.Dropout(0.2)(x)  # Regularize with dropout
        outputs = tfkl.Dense(1)(x)
        self.model = tfk.Model(inputs, outputs)

    def compile(self):
        """
        Compile the model
        """
        self.model.compile(
            loss=tfk.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tfk.optimizers.Adam(),
            metrics=[tfk.metrics.BinaryAccuracy()],
        )

    def train(self, X_train, y_train, X_val, y_val):
        """
        Performs model training
        """
        # Train the model and save its history
        self.history = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            # callbacks=[early_stopping],
        ).history
