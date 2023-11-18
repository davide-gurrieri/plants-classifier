from imports import *
from general_model import GeneralModel

build_param_1 = {
    "input_shape": (96, 96, 3),
    "output_shape": 2,
}

compile_param_1 = {
    "loss": tfk.losses.CategoricalCrossentropy(),
    "optimizer": tfk.optimizers.Adam(learning_rate=1e-3),
    "metrics": ["accuracy"],
}

compile_param_2 = {
    "loss": tfk.losses.CategoricalCrossentropy(),
    "optimizer": tfk.optimizers.Adam(learning_rate=5e-5),
    "metrics": ["accuracy"],
}

fit_param_1 = {
    "epochs": 200,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            mode="max",
            restore_best_weights=True,
        )
    ],
}

fit_param_2 = {
    "epochs": 200,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            mode="max",
            restore_best_weights=True,
        )
    ],
}


class ConvNeXtBaseCutmixMixup(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs, starting_model=None):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name
        
        self.base_model = tfk.applications.ConvNeXtBase(
            include_top=False,
            include_preprocessing=True,
            weights="imagenet",
            input_shape=build_kwargs["input_shape"],
            pooling="avg",
        )
        
        if starting_model is not None:
            self.model = starting_model
            self.compile()
            self.model.set_weights(starting_model.get_weights())

    def build(self):
        tf.random.set_seed(self.seed)

        augmentation = tf.keras.Sequential(
            [
                tfkl.RandomFlip("horizontal_and_vertical"),
                tfkl.RandomRotation(factor=0.3, fill_mode='reflect'),
                tfkl.RandomZoom(height_factor=-0.1),
            ],
            name="preprocessing",
        )

        relu_init = tfk.initializers.HeUniform(seed=self.seed)

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")

        augmentation_layer = augmentation(input_layer)
        
        self.base_model.trainable = False
        x = self.base_model(augmentation_layer)

        x = tfkl.Dropout(0.4)(x)

        x = tfkl.Dense(
            units=1000,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1L2(1e-3),
            kernel_initializer=relu_init,
        )(x)

        x = tfkl.Dropout(0.3)(x)

        x = tfkl.Dense(
            units=500,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1L2(1e-3),
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
    
    
