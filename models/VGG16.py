from imports import *
from general_model import GeneralModel
from tensorflow.keras.applications.vgg16 import (
    preprocess_input as preprocess_input_vgg16,
)

build_param_1 = {
    "input_shape": (96, 96, 3),
    "output_shape": 1,
}

compile_param_1 = {
    "loss": tfk.losses.BinaryCrossentropy(),
    "optimizer": tfk.optimizers.Adam(learning_rate=5e-4),
    "metrics": ["accuracy"],
}

fit_param_1 = {
    "batch_size": 32,
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


class VGG16(GeneralModel):
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

        preprocess_layer = preprocess_input_vgg16(augmentation_layer)

        # Build the VGG16
        VGG16_model = tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=self.build_kwargs["input_shape"],
            pooling="avg",
            classes=2,
            classifier_activation="sigmoid",
        )

        x = VGG16_model(preprocess_layer)


        # x = tfkl.Dropout(0.4)(x)

        # x = tfkl.Dense(
        #     units=1024,
        #     activation="relu",
        #     kernel_initializer=relu_init,
        # )(x)

        # x = tfkl.Dropout(0.3)(x)

        x = tfkl.Dense(
            units=512,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        # x = tfkl.Dropout(0.2)(x)

        x = tfkl.Dense(
            units=64,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        # x = tfkl.Dropout(0.1)(x)

        x = tfkl.Dense(
            units=56,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        # x = tfkl.Dropout(0.1)(x)

        output_layer = tfkl.Dense(
            units=self.build_kwargs["output_shape"],
            activation="sigmoid",
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.seed),
            name="Output",
        )(x)

        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name=self.name)
