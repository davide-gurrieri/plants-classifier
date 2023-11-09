from imports import *
from general_model import GeneralModel
# from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import preprocess_input

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


class MobileNet(GeneralModel):
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
                tfkl.RandomZoom(height_factor=0.3),
                # tfkl.RandomContrast(factor=0.8),
            ],
            name="preprocessing",
        )

        mobile = tfk.applications.MobileNetV2(
            input_shape=self.build_kwargs["input_shape"],
            include_top=False,                  # DON'T want INCLUDE the fully connected layer at the end of the network
            weights=None,                       # not loading the weights from the pretraining
            pooling='avg',
        )

        # Create an input layer with shape (96, 96, 3)
        input_layer = tfk.Input(shape=self.build_kwargs["input_shape"], name="Input")
        # Augmentation of the input
        augmentation_layer = augmentation(input_layer)
        # preprocessing layer
        preprocessing_layer = preprocess_input(augmentation_layer)
        # Connect MobileNetV2 to the input
        x = mobile(preprocessing_layer)
        # Adding dense layers
        classifier_layer = tfkl.Dense(units=1024, activation="relu", name="dense1")(flattening_layer)
        classifier_layer = tfkl.Dense(units=512, activation="relu", name="dense2")(classifier_layer)
        classifier_layer = tfkl.Dense(units=256, activation="relu", name="dense2")(classifier_layer)
        # Add a Dense layer with 1 units and sigmoid activation as the classifier
        output_layer = tfkl.Dense(
            units=self.build_kwargs["output_shape"],
            activation="sigmoid",
            name="Output",
        )(x)
        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name=self.name)

    