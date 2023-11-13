from imports import *
from general_model import GeneralModel
from tensorflow.keras.applications.xception import (
    preprocess_input as preprocess_input_xception,
)

build_param_1 = {
    "input_shape": (96, 96, 3),
    "output_shape": 1,
    "drop_0": 0.3,
    "drop_1": 0.3,
    "drop_2": 0.2,
    "drop_3": 0.1,
    "g_noise_0": 0.2,
    "g_noise_1": 0.2,
    "g_noise_2": 0.2,
    "g_noise_3": 0.1,
    "n_dense_1": 1024,
    "n_dense_2": 512,
    "n_dense_3": 64,
    "n_dense_4": 32,
}

compile_param_1 = {
    "loss": tfk.losses.BinaryCrossentropy(),
    "optimizer": tfk.optimizers.Adam(learning_rate=1e-4),
    "metrics": ["accuracy"],
}

fit_param_1 = {
    "batch_size": 64,
    "epochs": 2000,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            mode="max",
            restore_best_weights=True,
        )
    ],
}


class Xception(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def build(self):
        tf.random.set_seed(self.seed)

        input_shape = self.build_kwargs["input_shape"]
        output_shape = self.build_kwargs["output_shape"]
        relu_init = tfk.initializers.HeUniform(seed=self.seed)

        input_layer = tfkl.Input(shape=input_shape, name="Input")

        augmentation_layer = self.augmentation(input_layer)

        preprocessing_layer = preprocess_input_xception(augmentation_layer)

        base_model = tf.keras.applications.Xception(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
        # base_model.trainable = False

        # The base model contains batchnorm layers. We want to keep them in inference mode
        # when we unfreeze the base model for fine-tuning, so we make sure that the
        # base_model is running in inference mode here.
        x = base_model(preprocessing_layer)  # , training=False

        x = tfkl.Dropout(self.build_kwargs["drop_0"])(x)
        # x = tfkl.GaussianNoise(self.build_kwargs["g_noise_0"])(x)

        x = tfkl.Dense(
            units=self.build_kwargs["n_dense_1"],
            activation="relu",
            kernel_initializer=relu_init,
        )(x)
        x = tfkl.Dropout(self.build_kwargs["drop_1"])(x)
        x = tfkl.GaussianNoise(self.build_kwargs["g_noise_1"])(x)

        x = tfkl.Dense(
            units=self.build_kwargs["n_dense_2"],
            activation="relu",
            kernel_initializer=relu_init,
        )(x)
        x = tfkl.Dropout(self.build_kwargs["drop_2"])(x)
        x = tfkl.GaussianNoise(self.build_kwargs["g_noise_2"])(x)

        x = tfkl.Dense(
            units=self.build_kwargs["n_dense_3"],
            activation="relu",
            kernel_initializer=relu_init,
        )(x)
        x = tfkl.Dropout(self.build_kwargs["drop_3"])(x)
        x = tfkl.GaussianNoise(self.build_kwargs["g_noise_3"])(x)

        x = tfkl.Dense(
            units=self.build_kwargs["n_dense_4"],
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        output_layer = tfkl.Dense(
            units=output_shape,
            activation="sigmoid",
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.seed),
        )(x)
        self.model = tfk.Model(input_layer, output_layer, name=self.name)
