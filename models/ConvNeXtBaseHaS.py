from imports import *
from general_model import GeneralModel
# from tensorflow.keras.layers import Layer
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

class HideAndSeekLayer(tf.keras.layers.Layer):                  # polimorfismo sulla classe dei layer
  def __init__(self,
               hiding_prob,                                     # il metodo ha bisogno solo della probabilitá che una patch non sia visibile
               grid_h ):                                        # e della dimensione della griglia su cui costruire la mesh di patches

    super(HideAndSeekLayer, self).__init__()                    # eredito le proprietá dall'astrazione del layer
    self.hiding_prob = hiding_prob
    self.grid_size   = (grid_h,grid_h)
    self.training_mode = True

  def build(self, input_shape):                                 # Inizializzazione
    0                                                           # niente da inizializzare...


  def set_training_mode(self,val):
    self.training_mode = val
  def get_training_mode(self):
    return self.training_mode


  def call(self, inputs):                                       # Chiamata
    if not self.training_mode:
      return inputs
    mask = tf.random.uniform(
                      tf.tuple(
                      (len(inputs),
                       self.grid_size[0],
                       self.grid_size[1]
                       )
                       ) #considero la shape dell input privata della dimensione del colore
                  )

    # print(mask.shape)


    mask = (mask > self.hiding_prob)                           # nella maschera hiding prob descrive la probabilitá di oscuramento
    mask = tf.cast(mask, tf.float32)                           # ricasto a float

    mask = tf.image.resize(mask[:,:,:,None], inputs.shape[1 :-1], method = 'nearest')[:,:,:,0]


    mask = tf.concat([mask[:,:,:,None],
                      mask[:,:,:,None],
                      mask[:,:,:,None]], axis =3)                 # ri-introduco la dimensione del colore
    # print(mask.shape)



    return inputs * mask                                       # prodotto di Hadamard per oscurare
  
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
        
        hide_and_seek_layer = HideAndSeekLayer(0.5,8)

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")

        augmentation_layer = augmentation(input_layer)

        augmentation_layer = hide_and_seek_layer(augmentation_layer)

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
