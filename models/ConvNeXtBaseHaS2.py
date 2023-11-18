from imports import *
from general_model import GeneralModel, HideAndSeekLayer
# from tensorflow.keras.layers import Layer
import random

build_param_1 = {
    "input_shape": (96, 96, 3),
    "output_shape": 1,
}

compile_param_1 = {
    "loss": tfk.losses.BinaryCrossentropy(),
    "optimizer": tfk.optimizers.Adam(learning_rate=1e-4),
    "metrics": ["accuracy"],
}

compile_param_2 = {
    "loss": tfk.losses.BinaryCrossentropy(),
    "optimizer": tfk.optimizers.Adam(learning_rate=5e-5),
    "metrics": ["accuracy"],
}

fit_param_1 = {
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
  
class ConvNeXtBaseHaS(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs, starting_model=None):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name
        
        self.base_model = tfk.applications.ConvNeXtBase(
            include_top=False,
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

        # Create an ImageDataGenerator with the hide_patch function
        # datagen = ImageDataGenerator(preprocessing_function=hide_patch) 

        augmentation = tf.keras.Sequential(
            [
                tfkl.RandomFlip(mode="horizontal"),
                tfkl.RandomFlip(mode="vertical"),
                tfkl.RandomRotation(factor=0.25),
                tfkl.RandomZoom(height_factor=0.2),
            ],
            name="preprocessing",
        )

        relu_init = tfk.initializers.HeUniform(seed=self.seed)

        # possible grid size, 0 means no hiding
        grid_sizes = [6,8,12]
        # hiding probability
        hide_prob = 0.3

        hide_and_seek_layer = HideAndSeekLayer(hide_prob,grid_sizes)

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")

        augmentation_layer = augmentation(input_layer)

        augmentation_layer_1 = hide_and_seek_layer(augmentation_layer)

        preprocess_layer = tf.keras.applications.convnext.preprocess_input(augmentation_layer_1)
        
        self.base_model.trainable = False
        x = self.base_model(preprocess_layer)

        x = tfkl.Dropout(0.4)(x)

        x = tfkl.Dense(
            units=512,
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
