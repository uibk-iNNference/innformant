import numpy as np
import argparse

from resources import utils


def quantize(model_type: str, decimals: int) -> str:
    model = utils.get_model(model_type)
    new_model_type = f"{model_type}-quantized-{decimals}"

    for layer in model.layers:
        weights = layer.get_weights()
        for i, weight in enumerate(weights):
            np.around(weight, decimals, out=weights[i])
        layer.set_weights(weights)

    utils.save_model(new_model_type, model)
    return new_model_type


def dump(model_type: str):
    model = utils.get_model(model_type)

    for layer in model.layers:
        for weight in layer.get_weights():
            print(weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_type", help="The name of the model, all layers will be quantized")
    parser.add_argument(
        "--decimals", type=int, default=7, help="The number of decimals to round to (default: 7)")
    parser.add_argument(
        "--dump", action="store_true", help="Reload model weights after quantizing and dump")

    args = parser.parse_args()
    new_name = quantize(args.model_type, args.decimals)
    if args.dump:
        dump(new_name)
