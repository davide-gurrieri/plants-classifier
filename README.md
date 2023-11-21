# Politecnico di Milano - Image Classification challenge for Artificial Neural Network and Deep Learning course


This is the model used in the [Image Classification challenge](https://codalab.lisn.upsaclay.fr/competitions/16245
) as part of the "Artificial Neural Networks and Deep Learning" course at Politecnico di Milano in 2023/24.
The goal of the challenge was to classify plants health status from photographs of their leaves that were 96x96 pixels with 3 color channels. We employed a transfer learning approach, using ConvNeXt as the base model.

This model let us reach an accuracy of 86.80% ([Results](https://codalab.lisn.upsaclay.fr/competitions/16245#results)).

## Model details

For more details about the model check the [Report](https://github.com/davide-gurrieri/plants-classifier/blob/main/report/Report_AN2DL_Challenge1.pdf). Briefly, we used the following techniques:
* Transfer Learning and Fine Tuning with ConvNeXtLarge model with the "Weight Initialization" technique
* Standard data augmentation available in Keras (flip, rotation, zoom)
* CutMix and MixUp data augmentation techniques available in Keras-CV: one or the other technique was applied to each input image
* Early Stopping Callback in order to understand the optimal number of epochs to use in the training phase
* Ensemble: element-wise maximum strategy of two models 

## Set up
1. Download the dataset from [here](https://drive.google.com/file/d/1llWCmIbaW-uHvZcD-soT8DJQJYmm8zAA/view?usp=sharing) and save the zip inside the data folder.
2. Run the model
