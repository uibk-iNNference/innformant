from enum import Enum
import functools

import tensorflow.keras as keras
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
import itertools
from resources import config as cfg


def normalize_img(num_classes, image, label):
    image = tf.cast(image, tf.float32) / 255.
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def preprocess(dataset, info, batch_size=32, shuffle=True):
    normalize_function = functools.partial(
        normalize_img, info.features['label'].num_classes)

    dataset = dataset.map(
        normalize_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(len(dataset), seed=42)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def get_single_test_sample(dataset, index=42, include_label=False):
    #discard everything before our desired index
    dataset = dataset.skip(index)

    for sample, label in dataset:
        if include_label:
            return sample.numpy(), label.numpy()
        else:
            return sample.numpy()


def get_mnist_data(batch_size=32):
    (train_ds, test_ds), info = tfds.load('mnist',
                                          split=['train', 'test'],
                                          as_supervised=True,
                                          with_info=True,
                                          data_dir=cfg.get_data_dir(),
                                          )
    train_ds = preprocess(train_ds, info, batch_size=batch_size, shuffle=True)
    test_ds = preprocess(test_ds, info, batch_size=batch_size, shuffle=False)
    return train_ds, test_ds


def get_single_mnist_test_sample(index=42, include_label=False):
    _, test_ds = get_mnist_data(batch_size=1)
    return get_single_test_sample(test_ds, index, include_label)


def get_cifar10_data(batch_size=32):
    (train_ds, test_ds), info = tfds.load('cifar10',
                                          split=['train', 'test'],
                                          as_supervised=True,
                                          with_info=True,
                                          data_dir=cfg.get_data_dir(),
                                          )
    train_ds = preprocess(train_ds, info, batch_size=batch_size, shuffle=True)
    test_ds = preprocess(test_ds, info, batch_size=batch_size, shuffle=False)
    return train_ds, test_ds


def get_single_cifar10_test_sample(index=42, include_label=False):
    _, test_ds = get_cifar10_data(batch_size=1)
    return get_single_test_sample(test_ds, index, include_label)


def get_fmnist_data(batch_size=32):
    (train_ds, test_ds), info = tfds.load('fashion_mnist',
                                          split=['train', 'test'],
                                          as_supervised=True,
                                          with_info=True,
                                          data_dir=cfg.get_data_dir(),
                                          )
    train_ds = preprocess(train_ds, info, batch_size=batch_size, shuffle=True)
    test_ds = preprocess(test_ds, info, batch_size=batch_size, shuffle=False)
    return train_ds, test_ds


def get_single_fmnist_test_sample(index=42, include_label=False):
    _, test_ds = get_fmnist_data(batch_size=1)
    return get_single_test_sample(test_ds, index, include_label)


def get_imagenet_data(batch_size=32, preprocess=False):
    try:
        return load_imagenet(batch_size, preprocess)
    except AssertionError:
        download_imagenet()
        return load_imagenet(batch_size, preprocess)


def load_imagenet(batch_size, preprocess=False):
    validation_ds, _ = tfds.load('imagenet2012',
                                 split='validation',
                                 data_dir=cfg.get_data_dir(),
                                 as_supervised=True,
                                 with_info=True,
                                 batch_size=batch_size
                                 )
    if preprocess:
        validation_ds = validation_ds.map(
            lambda x, y: (tf.image.resize(x, (224, 224)), y))
        validation_ds = validation_ds.map(lambda x, y:
                                          (keras.applications.resnet_v2.preprocess_input(x),
                                           tf.one_hot(y, 1000)))
    return validation_ds


def get_single_imagenet_test_sample(index=42, include_label=False):
    validation_ds = get_imagenet_data(preprocess=True)

    return get_single_test_sample(validation_ds, index, include_label)

def download_imagenet():
    filename = os.path.join(cfg.get_data_dir(),
                            "downloads",
                            "manual",
                            cfg.IMAGENET_DATASET_FILENAME)
    filename = os.path.abspath(os.path.expanduser(filename))

    dl_manager = tfds.download.DownloadManager(
        download_dir=cfg.get_data_dir(),
        extract_dir=cfg.get_data_dir())
    resource = tfds.download.Resource(url=cfg.IMAGENET_DOWNLOAD_URL)
    downloaded_file = dl_manager.download(resource)
    downloaded_file = os.path.abspath(downloaded_file)
    if not resource.exists_locally(downloaded_file):
        raise SystemError(f"Download from {resource.url} failed")

    os.makedirs(os.path.join(
        cfg.get_data_dir(),
        "manual"), exist_ok=True)
    os.rename(downloaded_file, filename)
