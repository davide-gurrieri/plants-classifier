from imports import *
from general_model import GeneralModel

build_param_1 = {
    "input_shape": (96, 96, 3),
    "output_shape": 2,
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
    "loss": tfk.losses.CategoricalCrossentropy(),
    "optimizer": tfk.optimizers.Adam(learning_rate=1e-4),
    "metrics": ["accuracy"],
}

fit_param_1 = {
    "batch_size": 128,
    "epochs": 2000,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
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

        scale_layer = tfkl.Rescaling(scale=1 / 127.5, offset=-1)(input_layer)

        augmentation = self.augmentation

        preprocessing_layer = augmentation(scale_layer)

        base_model = tfk.applications.Xception(
            weights="imagenet",
            input_shape=input_shape,
            include_top=False,
        )
        base_model.trainable = False

        # The base model contains batchnorm layers. We want to keep them in inference mode
        # when we unfreeze the base model for fine-tuning, so we make sure that the
        # base_model is running in inference mode here.
        x = base_model(preprocessing_layer, training=False)

        x = tfkl.GlobalAveragePooling2D()(x)

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
            activation="softmax",
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.seed),
        )(x)
        self.model = tfk.Model(input_layer, output_layer, name=self.name)
