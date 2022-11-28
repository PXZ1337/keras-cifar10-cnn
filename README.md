# Convolutional Neural Network trained on the CIFAR10-Dataset

This example shows how to train a convolutinal neural network on the cifar10 dataset. I tried to compare different models and experiments to get the best possible performance on accuracy with minimal effort.

## Possible improvements of the network

- Using random search without cross validation to determine good hyper parameter
- Using dropout rates to prevent overfitting
- Use batch normalization
- Adapt learning rates with different lr scheduler functions
- Use a lr plateau callback to adjust lr by training progess
- Use early stopping callbacks to prevent unnecessary training epochs
- Use global pooling instead of just flattening
- Try and error

## Tensorboard

The performance of the network is analyzed through the tensorboard application.

Run `tensorboard --logdir=./src/logs` to start tensorboard and compare the model performance.

## About the dataset

This dataset contains 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories.

## Classes

|Label|Description|
|-----|-----------|
|0|airplane|
|1|automobile|
|2|bird|
|3|cat|
|4|deer|
|5|dog|
|6|frog|
|7|horse|
|8|ship|
|9|truck|
