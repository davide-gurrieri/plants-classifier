from imports import *
from general_model import GeneralModel

build_param_1 = {
    "input_shape": (96, 96, 3),
    "output_shape": 1,
    "filters_1": 64,
    "filters_2": 128,
    "filters_3": 256,
    "filters_4": 512,
    "kernel_size": 3,
    "stack": 2,
}

compile_param_1 = {
    "loss": tfk.losses.BinaryCrossentropy(),
    "optimizer": tfk.optimizers.Adam(learning_rate=0.001, weight_decay=5e-4),
    "metrics": ["accuracy"],
}

fit_param_1 = {
    "batch_size": 32,
    "epochs": 500,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            mode="max",
            restore_best_weights=True,
        ),
        tfk.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.1, patience=20, min_lr=1e-5, mode="max"
        ),
    ],
}


class VGG18Residual(GeneralModel):
    """
    VGG with Batch Normalization and Skip Connection.

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

    name = ""

    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def conv_residual_block(
        self,
        x,
        filters,
        kernel_size,
        padding="same",
        downsample=True,
        activation="relu",
        stack=None,
        batch_norm=True,
        name="",
    ):
        """
        Definition of a residual convolutional block with optinal batch normalization
        """

        stack = self.build_kwargs["stack"]

        if downsample:
            x = tfkl.MaxPooling2D(name="MaxPool_" + name)(x)

        x_ = x

        for s in range(stack):
            x_ = tfkl.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                name="Conv_" + name + str(s + 1),
            )(x_)
            if batch_norm:
                x_ = tfkl.BatchNormalization(name="BatchNorm_" + name + str(s + 1))(x_)
            x_ = tfkl.Activation(activation, name="Activation_" + name + str(s + 1))(x_)

        if downsample:
            x = tfkl.Conv2D(
                filters=filters,
                kernel_size=1,
                padding=padding,
                name="Conv_" + name + "skip",
            )(x)

        x = tfkl.Add(name="Add_" + name)([x_, x])

        return x

    def build(self):
        """
        Definition and building of the model
        """

        tf.random.set_seed(self.seed)

        # Build the neural network layer by layer

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")

        scale_layer = tfkl.Rescaling(scale=1 / 255, offset=0)(input_layer)

        preprocessing = super().augmentation(scale_layer)

        x = tfkl.Conv2D(filters=64, kernel_size=3, padding="same", name="Conv0")(
            preprocessing
        )
        x = tfkl.BatchNormalization(name="BatchNorm0")(x)
        x = tfkl.Activation("relu", name="ReLU0")(x)

        x = self.conv_residual_block(
            x=x,
            filters=self.build_kwargs["filters_1"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            name="1",
        )
        x = self.conv_residual_block(
            x=x,
            filters=self.build_kwargs["filters_1"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            name="2",
        )

        x = self.conv_residual_block(
            x=x,
            filters=self.build_kwargs["filters_2"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=True,
            name="3",
        )
        x = self.conv_residual_block(
            x=x,
            filters=self.build_kwargs["filters_2"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            name="4",
        )

        x = self.conv_residual_block(
            x=x,
            filters=self.build_kwargs["filters_3"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=True,
            name="5",
        )
        x = self.conv_residual_block(
            x=x,
            filters=self.build_kwargs["filters_3"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            name="6",
        )

        x = self.conv_residual_block(
            x=x,
            filters=self.build_kwargs["filters_4"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=True,
            name="7",
        )
        x = self.conv_residual_block(
            x=x,
            filters=self.build_kwargs["filters_4"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            name="8",
        )

        x = tfkl.GlobalAveragePooling2D(name="GlobalAveragePooling")(x)
        # x = tfkl.Dense(self.build_kwargs["output_shape"], name='Dense')(x)
        # output_activation = tfkl.Activation('softmax', name='Softmax')(x)
        output_layer = tfkl.Dense(
            units=self.build_kwargs["output_shape"], activation="sigmoid", name="Output"
        )(x)

        # Connect input and output through the Model class
        self.model = tfk.Model(
            inputs=input_layer, outputs=output_layer, name="VGG18_Residual"
        )
