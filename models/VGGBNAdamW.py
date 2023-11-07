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
    "optimizer": tfk.optimizers.AdamW(learning_rate=0.001,weight_decay=5e-4),
    "metrics": ["accuracy"],
}

fit_param_1 = {
    "batch_size": 128,
    "epochs": 500,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=25, mode="max", restore_best_weights=True,
        ),
        tfk.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.1, patience=20, min_lr=1e-5, mode="max"
        ),
    ],
}


class VGGBNAdamW(GeneralModel):
    """
    First simple model copied from the lab.

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

    def conv_block(
        self,
        x,
        filters,
        kernel_size,
        padding="same",
        downsample=True,
        activation="relu",
        stack=None,
        name="",
    ):

        stack = self.build_kwargs["stack"]

        # If downsample is True, apply max-pooling
        if downsample:
            x = tfkl.MaxPooling2D(name="MaxPool_" + name)(x)

        # Apply a stack of convolutional layers with specified filters, kernel size, and activation
        for s in range(stack):
            x = tfkl.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                name="Conv_" + name + str(s + 1),
            )(x)
            x = tfkl.Activation(activation, name="Activation_" + name + str(s + 1))(x)

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

        # Initial convolution and activation
        x0 = tfkl.Conv2D(filters=64, kernel_size=3, padding="same", name="Conv0")(
            preprocessing
        )
        x0 = tfkl.Activation("relu", name="ReLU0")(preprocessing)

        # Create convolutional blocks
        x1 = self.conv_block(
            x=x0,
            filters=self.build_kwargs["filters_1"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            stack=self.build_kwargs["stack"],
            name="1",
        )
        x1 = self.conv_block(
            x=x0,
            filters=self.build_kwargs["filters_1"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            stack=self.build_kwargs["stack"],
            name="2",
        )

        x2 = self.conv_block(
            x=x0,
            filters=self.build_kwargs["filters_2"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            stack=self.build_kwargs["stack"],
            name="3",
        )
        x2 = self.conv_block(
            x=x0,
            filters=self.build_kwargs["filters_2"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            stack=self.build_kwargs["stack"],
            name="4",
        )

        x3 = self.conv_block(
            x=x0,
            filters=self.build_kwargs["filters_3"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            stack=self.build_kwargs["stack"],
            name="5",
        )
        x3 = self.conv_block(
            x=x0,
            filters=self.build_kwargs["filters_3"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            stack=self.build_kwargs["stack"],
            name="6",
        )

        x4 = self.conv_block(
            x=x0,
            filters=self.build_kwargs["filters_4"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            stack=self.build_kwargs["stack"],
            name="7",
        )
        x4 = self.conv_block(
            x=x0,
            filters=self.build_kwargs["filters_4"],
            kernel_size=self.build_kwargs["kernel_size"],
            downsample=False,
            stack=self.build_kwargs["stack"],
            name="8",
        )

        # Global Average Pooling and classifier
        x = tfkl.GlobalAveragePooling2D(name="GlobalAveragePooling")(x4)
        output_layer = tfkl.Dense(
            self.build_kwargs["output_shape"], activation="sigmoid", name="Dense"
        )(x)

        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name="Convnet")
    
        
