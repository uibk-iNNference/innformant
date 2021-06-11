import os

from tensorflow.keras.models import Sequential, load_model
import numpy as np

from resources import config as cfg

models = {}


def ensure_prediction_directory():
    os.makedirs(cfg.get_prediction_directory(), exist_ok=True)


def get_model(model_type: str) -> Sequential:
    if model_type not in models.keys():
        models[model_type] = load_model(os.path.join(
            cfg.get_weight_dir(), f"{model_type}.h5"))
        models[model_type].compile()
    return models[model_type]


def get_output_shape(model_type: str) -> tuple:
    return get_model(model_type).output_shape


def save_model(model_type: str, model: Sequential) -> None:
    filename = os.path.join(cfg.get_weight_dir(), f"{model_type}.h5")
    return model.save(filename)


def get_class(value: np.ndarray, unique_values: list) -> int:
    candidate_class = 0
    # test if any of the existing classes matches the value
    for unique_value in unique_values:
        if np.all(value == unique_value):
            break
        candidate_class += 1

    # if none do, append the value
    if candidate_class == len(unique_values):
        unique_values.append(value)

    return candidate_class


def calculate_psnr(x, x_adv, max_val=1):
    if np.all(x == x_adv):
        return 0
    difference = x - x_adv

    mean_squared_error = np.mean(np.square(difference))
    psnr = 10 * np.log10(max_val*max_val / mean_squared_error)
    return psnr
