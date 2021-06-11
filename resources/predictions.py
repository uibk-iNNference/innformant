from typing import Union

import numpy as np
import tensorflow.keras.losses as losses
from tensorflow import GradientTape, constant

from resources import utils


def predict(model_type: str, sample: np.ndarray):
    model = utils.get_model(model_type)
    return model(sample).numpy()


def predict_with_gradient(model_type: str, sample: np.ndarray, label: np.ndarray,
                          loss: Union[losses.Loss, str] = losses.CategoricalCrossentropy()):
    model_type = model_type.lower()
    model = utils.get_model(model_type)
    model.trainable = False

    if isinstance(loss, str):
        loss = losses.get(loss)

    sample_t = constant(sample)

    with GradientTape() as tape:
        # explicitly add input tensor to tape
        tape.watch(sample_t)
        # get prediction
        prediction = model(sample_t)
        # calculate loss
        loss_value = loss(label, prediction)

    # calculate dloss/dx
    gradients = tape.gradient(loss_value, sample_t)
    return prediction.numpy(), gradients.numpy()


def predict_full_dataset(model_type: str):

    model = utils.get_model(model_type)

    return predictions
