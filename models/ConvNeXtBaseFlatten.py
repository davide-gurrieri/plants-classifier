from imports import *
from general_model import GeneralModel

build_param_1 = {
    "input_shape": (96, 96, 3),
    "output_shape": 1,
}

compile_param_1 = {
    "loss": tfk.losses.BinaryCrossentropy(),
    "optimizer": tfk.optimizers.Adam(learning_rate=1e-4),
    "metrics": ["accuracy"],
}

fit_param_1 = {
    "batch_size": 32,
    "epochs": 200,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            mode="max",
            restore_best_weights=True,
        )
    ],
}


class ConvNeXtBase(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def build(self):
        tf.random.set_seed(self.seed)

        augmentation = tf.keras.Sequential(
            [
                tfkl.RandomFlip(mode="horizontal"),
                tfkl.RandomFlip(mode="vertical"),
                tfkl.RandomRotation(factor=0.25),
                # tfkl.RandomCrop(height=64, width=64),
                tfkl.RandomZoom(height_factor=0.2),
                # tfkl.RandomContrast(factor=0.8),
            ],
            name="preprocessing",
        )

        relu_init = tfk.initializers.HeUniform(seed=self.seed)

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")

        augmentation_layer = augmentation(input_layer)

        # Build the ResNet50
        ConvNeXtBase = tfk.applications.ConvNeXtBase(
            include_top=False,
            include_preprocessing=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=self.build_kwargs["input_shape"],
            classes=2,
            pooling=None,
        )

        x = ConvNeXtBase(augmentation_layer)

        x = tfkl.Flatten()(x)

        x = tfkl.Dropout(0.4)(x)

        x = tfkl.Dense(
            units=1024,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        x = tfkl.Dropout(0.3)(x)

        x = tfkl.Dense(
            units=512,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        x = tfkl.Dropout(0.2)(x)

        x = tfkl.Dense(
            units=64,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        x = tfkl.Dropout(0.1)(x)

        x = tfkl.Dense(
            units=56,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        x = tfkl.Dropout(0.1)(x)

        output_layer = tfkl.Dense(
            units=self.build_kwargs["output_shape"],
            activation="sigmoid",
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.seed),
            name="Output",
        )(x)

        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name=self.name)
