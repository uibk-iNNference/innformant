import argparse
import os
import numpy as np

from resources import config as cfg, utils, predictions, data


def predict_individual(model_type: str):
    model_type = model_type.lower()
    if model_type == 'mnist':
        sample = data.get_single_mnist_test_sample()
    elif model_type == 'cifar10':
        sample = data.get_single_cifar10_test_sample()
    elif model_type == 'fmnist':
        sample = data.get_single_fmnist_test_sample()
    elif model_type == 'imagenet':
        sample = data.get_single_imagenet_test_sample()
    else:
        raise RuntimeError(f"Unknown model type {model_type}")

    print(f"Predicting single sample for model {model_type}")
    prediction_result = predictions.predict(model_type, sample)

    utils.ensure_prediction_directory()
    target_filename = os.path.join(
        cfg.get_prediction_directory(),
        cfg.FILE_TEMPLATE.format(
            model_type)
    )
    np.save(target_filename, prediction_result)


def predict_full_dataset(model_type: str):
    model_type = model_type.lower()
    if model_type == 'mnist':
        _, samples = data.get_mnist_data(batch_size=1)
    elif model_type == 'fmnist':
        _, samples = data.get_fmnist_data(batch_size=1)
    elif model_type == 'cifar10':
        _, samples = data.get_cifar10_data(batch_size=1)
    elif 'imagenet' in model_type:
        samples = data.get_imagenet_data(batch_size=1, preprocess=True)
    else:
        raise SystemError(f"Unknown model type {model_type}")

    print(f"Predicting full dataset for model {model_type}")
    num_samples = cfg.NUM_SAMPLES
    prediction_results = np.empty(
        (num_samples,) + utils.get_output_shape(model_type)[1:])

    # reduce dataset to our desired size
    samples = samples.take(cfg.NUM_SAMPLES)

    for i, (sample, _) in enumerate(samples):
        if i % 100 == 0:
            print(f"Sample {i}")
        prediction_results[i] = predictions.predict(model_type, sample)

    utils.ensure_prediction_directory()
    target_filename = os.path.join(
        cfg.get_prediction_directory(),
        cfg.FILE_TEMPLATE.format(
            f"{model_type}_full")
    )
    np.save(target_filename, prediction_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Generate predictions")
    parser.add_argument("model_type", type=str,
                        help="The model type to use")
    parser.add_argument("--full", action="store_true",
                        default=False, help="Predict full dataset (reduced to 500 samples for now")
    args = parser.parse_args()

    if args.full:
        predict_full_dataset(args.model_type)
    else:
        predict_individual(args.model_type)
