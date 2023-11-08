from imports import *
from general_model import GeneralModel

build_param_1 = {
    "input_shape": (96, 96, 3),
    "output_shape": 2,
}

compile_param_1 = {
    "loss": tfk.losses.CategoricalCrossentropy(),
    "optimizer": tfk.optimizers.Adam(learning_rate=5e-4),
    "metrics": ["accuracy"],
}

fit_param_1 = {
    "batch_size": 128,
    "epochs": 200,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            mode="max",
            restore_best_weights=True,
        )
    ],
}


class QuasiVGG9(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def build(self):
        tf.random.set_seed(self.seed)

        relu_init = tfk.initializers.HeUniform(seed=self.seed)

        # Build the neural network layer by layer
        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")

        scale_layer = tfkl.Rescaling(scale=1 / 255, offset=0)(input_layer)

        preprocessing_layer = self.augmentation(scale_layer)

        x = tfkl.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=relu_init,
            name="conv00",
        )(preprocessing_layer)
        x = tfkl.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=relu_init,
            name="conv01",
        )(x)
        x = tfkl.MaxPooling2D(name="mp0")(x)

        x = tfkl.Conv2D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=relu_init,
            name="conv10",
        )(x)
        x = tfkl.Conv2D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=relu_init,
            name="conv11",
        )(x)
        x = tfkl.MaxPooling2D(name="mp1")(x)

        x = tfkl.Conv2D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=relu_init,
            name="conv20",
        )(x)
        x = tfkl.Conv2D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=relu_init,
            name="conv21",
        )(x)
        x = tfkl.MaxPooling2D(name="mp2")(x)

        x = tfkl.Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=relu_init,
            name="conv30",
        )(x)
        x = tfkl.Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=relu_init,
            name="conv31",
        )(x)

        # ctrl + k + c
        # ctrl + k + u
        x = tfkl.MaxPooling2D(name="mp3")(x)

        x = tfkl.Conv2D(
            filters=512,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=relu_init,
            name="conv40",
        )(x)
        x = tfkl.Conv2D(
            filters=512,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=relu_init,
            name="conv41",
        )(x)
        x = tfkl.GlobalAveragePooling2D(name="gap")(x)

        x = tfkl.Dense(
            units=1024,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        x = tfkl.Dense(
            units=512,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        x = tfkl.Dense(
            units=256,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        output_layer = tfkl.Dense(
            units=self.build_kwargs["output_shape"],
            activation="softmax",
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.seed),
            name="Output",
        )(x)

        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name=self.name)
