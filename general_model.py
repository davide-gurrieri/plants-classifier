from imports import *


class GeneralModel:
    name = ""
    seed = SEED
    model = tfk.Model()
    history = {}
    history_val = {}
    cv_n_fold = 0
    cv_histories = []
    cv_scores = []
    cv_best_epochs = []
    cv_avg_epochs = -1
    augmentation = tf.keras.Sequential(
        [
            tfkl.RandomFlip(mode="horizontal"),
            tfkl.RandomFlip(mode="vertical"),
            tfkl.RandomRotation(factor=0.25),
            # tfkl.RandomContrast(factor=0.8),
        ],
        name="preprocessing",
    )
    base_model = None

    def __init__(self, build_kwargs={}, compile_kwargs={}, fit_kwargs={}):
        self.build_kwargs = build_kwargs
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs

    def build(self):
        pass

    def compile(self):
        """
        Compile the model
        """
        self.model.compile(**self.compile_kwargs)

    def train(
        self,
        x_train=None,
        y_train=None,
    ):
        """
        Train the model
        """
        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            **self.fit_kwargs,
        ).history

    def train_val(
        self,
        x_train=None,
        y_train=None,
        x_val=None,
        y_val=None,
    ):
        """
        Train the model
        """
        validation_data = x_val if y_val is None else (x_val, y_val)
        self.history_val = self.model.fit(
            x=x_train,
            y=y_train,
            validation_data=validation_data,
            **self.fit_kwargs,
        ).history

    def save_model(self):
        """
        Save the trained model in the models folder
        """
        self.model.save(f"saved_models/{self.name}")

    def plot_history(self, training=True, figsize=(15, 2)):
        """
        Plot the loss and metrics for the training and validation sets with respect to the training epochs.

        Parameters
        ----------
        training : bool, optional
            show the training plots, by default True
        figsize : tuple, optional
            dimension of the plots, by default (15, 2)
        """
        keys = list(self.history_val.keys())
        n_metrics = len(keys) // 2

        for i in range(n_metrics):
            plt.figure(figsize=figsize)
            if training:
                plt.plot(
                    self.history_val[keys[i]], label="Training " + keys[i], alpha=0.8
                )
            plt.plot(
                self.history_val[keys[i + n_metrics]],
                label="Validation " + keys[i],
                alpha=0.8,
            )
            plt.title(keys[i])
            plt.legend()
            plt.grid(alpha=0.3)

        plt.show()
        
    def predict(self, x_eval):
        predictions = self.model.predict(x_eval, verbose=0)
        if self.build_kwargs["output_shape"] == 2:
            predictions = np.argmax(predictions, axis=-1)
        else:
            predictions = np.squeeze(predictions, axis=1)
            predictions = np.ndarray.round(predictions)
        predictions = predictions.astype(int)
        return predictions
        
        

    def evaluate(self, x_eval, y_eval):
        """
        Evaluate the model on the evaluation set.

        Parameters
        ----------
        X_eval : numpy.ndarray
            Evaluation input data
        y_eval : numpy.ndarray
            Evaluation target data
        """
        predictions = self.predict(x_eval)
        accuracy = accuracy_score(y_eval, predictions)
        precision = precision_score(y_eval, predictions)
        recall = recall_score(y_eval, predictions)
        cm = confusion_matrix(y_eval, predictions)
        print(
            f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nConfusion matrix:\n{cm}"
        )
        return accuracy
    
    def print_base_model(self):
        for i, layer in enumerate(self.model.get_layer(self.base_model.name).layers):
            print(i, layer.name, layer.trainable)
        
    def unfreeze_layers(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.model.get_layer(self.base_model.name).layers)
        self.model.get_layer(self.base_model.name).trainable = True
        for layer in self.model.get_layer(self.base_model.name).layers[start:end]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable=True
        self.compile()
    

class HideAndSeekLayer(tf.keras.layers.Layer):                  # polimorfismo sulla classe dei layer
  def __init__(self,
               hiding_prob,                                     # il metodo ha bisogno solo della probabilitá che una patch non sia visibile
               possible_sizes):                                        # e della dimensione della griglia su cui costruire la mesh di patches

    super(HideAndSeekLayer, self).__init__()                    # eredito le proprietá dall'astrazione del layer
    self.hiding_prob = hiding_prob
    self.possible_sizes   = possible_sizes
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
    grid_size= self.possible_sizes[random.randint(0,len(self.possible_sizes)-1)]
    mask = tf.random.uniform(
                      tf.tuple(
                      (len(inputs),
                       grid_size,
                       grid_size
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
