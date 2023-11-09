from imports import *
from general_model import GeneralModel
from tensorflow.keras.applications.resnet50 import preprocess_input

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


class ResNet50(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def build(self):
        tf.random.set_seed(self.seed)
        
        relu_init = tfk.initializers.HeUniform(seed=self.seed)

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")
        
        augmentation = self.augmentation(input_layer)
        
        preprocess_layer= preprocess_input(augmentation)

        # Build the ResNet50
        ResNet50=tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=self.build_kwargs["input_shape"],
            pooling="avg",
        )
        
        x =ResNet50(preprocess_layer)

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

        x = tfkl.Dense(
            units=128,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        x = tfkl.Dense(
            units=32,
            activation="relu",
            kernel_initializer=relu_init,
        )(x)

        output_layer = tfkl.Dense(
            units=self.build_kwargs["output_shape"],
            activation="sigmoid",
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.seed),
            name="Output",
        )(x)

        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name=self.name)
