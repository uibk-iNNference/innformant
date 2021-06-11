# Python RNG
import argparse

from tensorflow.keras.optimizers import Adam
from resources import data, utils, models
from tensorflow.python.framework import random_seed
import numpy as np
import random

random.seed(42)
# Numpy RNG

np.random.seed(42)
# TF RNG

random_seed.set_seed(42)


def train(model_type, model, train_ds, test_ds, epochs, learning_rate=0.001):
    model.compile(optimizer=Adam(learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds,
              epochs=epochs, validation_data=test_ds, validation_steps=1)
    score = model.evaluate(test_ds, verbose=0)

    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")

    print("Saving weights...")
    utils.save_model(model_type, model)


def main(model_type):
    model_type = model_type.lower()

    if model_type == 'mnist':
        print("Training mnist")
        train_ds, test_ds = data.get_mnist_data(batch_size=128)
        model = models.build_mnist_cnn()
        train(model_type, model, train_ds, test_ds, epochs=5)

    elif model_type == 'mnist_mlp':
        print("Training mnist_mlp")
        train_ds, test_ds = data.get_mnist_data(batch_size=128)
        model = models.build_mnist_mlp()
        train(model_type, model, train_ds, test_ds, epochs=5)

    elif model_type == 'single_layer':
        print("Training single layer")
        model = models.build_single_layer()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("Saving weights...")
        utils.save_model(model_type, model)

    elif model_type == 'min_conv_layer_1x1':
        print("Training single layer")
        model = models.build_min_conv_layer(kernel_size=1)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("Saving weights...")
        utils.save_model(model_type, model)

    elif model_type.startswith('min_conv_layer'):
        model_conv = model_type.replace('min_conv_layer_', '').split('_')
        model_conv = list(map(int, model_conv))

        print(
            f"Training minimum conv layer with {model_conv[0]} filters and kernel size {model_conv[1]}")
        model = models.build_min_conv_layer(
            filters=model_conv[0], kernel_size=model_conv[1])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("Saving weights...")
        utils.save_model(model_type, model)

    elif model_type == 'single_conv_layer':
        print("Training single conv layer")
        model = models.build_single_conv_layer()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("Saving weights...")
        utils.save_model(model_type, model)

    elif model_type == 'cifar10':
        print(f"Training cifar10")

        train_ds, test_ds = data.get_cifar10_data(batch_size=128)
        model = models.build_cifar10()
        train(model_type, model, train_ds, test_ds, epochs=150)

    elif model_type == 'fmnist':
        print(f"Training fmnist")

        train_ds, test_ds = data.get_fmnist_data(batch_size=128)
        model = models.build_fmnist()
        train(model_type, model, train_ds, test_ds, epochs=150)

    elif model_type == 'imagenet':
        model = models.get_pretrained_imagenet('imagenet')
        print(f"Training imagenet")

        validation_ds = data.get_imagenet_data(batch_size=16, preprocess=True)
        train(model_type, model, validation_ds, validation_ds, epochs=1, learning_rate=0.000001)

    else:
        raise RuntimeError(f"Unknown model type {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model used for evaluation")
    parser.add_argument("model_type", type=str, help="One of {'mnist', 'mnist_mlp', 'single_layer', 'cifar10', "
                                                     "'fmnist'}; the type of model to train")

    args = parser.parse_args()
    main(args.model_type)
