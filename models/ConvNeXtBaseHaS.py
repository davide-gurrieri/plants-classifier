from imports import *
from general_model import GeneralModel
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

class HidePatchLayer(Layer):
    def __init__(self, hide_prob=0.5, grid_sizes=[0, 4, 8, 16, 32], **kwargs):
        super(HidePatchLayer, self).__init__(**kwargs)
        self.hide_prob = hide_prob
        self.grid_sizes = grid_sizes

    def call(self, inputs, training=None):
        if training:
            img = tf.convert_to_tensor(inputs)

            img = tf.identity(inputs)
            # img_aug = np.copy(img)
            # s = img_aug.shape
            s = img.shape

            wd, ht = s[1], s[2]

            # randomly choose one grid size
            grid_size = self.grid_sizes[random.randint(0, len(self.grid_sizes) - 1)]

            # hide the patches
            if grid_size != 0:
                for x in range(0, wd, grid_size):
                    for y in range(0, ht, grid_size):
                        x_end = min(wd, x + grid_size)
                        y_end = min(ht, y + grid_size)
                        if random.random() <= self.hide_prob:
                            img = tf.tensor_scatter_nd_update(
                                img,
                                tf.where(tf.ones_like(img[..., :1])),
                                tf.zeros_like(img[..., :1])
                            )
            return img
        else:
            return inputs


class ConvNeXtBaseHaS(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def build(self):
        
        tf.random.set_seed(self.seed)

        # Create an ImageDataGenerator with the hide_patch function
        # datagen = ImageDataGenerator(preprocessing_function=hide_patch) 

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
        
        hide_patch_layer = HidePatchLayer()

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")

        augmentation_layer = augmentation(input_layer)

        augmentation_layer = hide_patch_layer(augmentation_layer, training=True)

        # Build the ConvNeXtBase
        ConvNeXtBase = tfk.applications.ConvNeXtBase(
            include_top=False,
            include_preprocessing=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=self.build_kwargs["input_shape"],
            classes=2,
            pooling="avg",
        )

        x = ConvNeXtBase(augmentation_layer)

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
