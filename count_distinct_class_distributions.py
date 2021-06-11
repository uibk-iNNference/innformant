import sys

import numpy as np
import argparse
import glob

from prettytable import PrettyTable, MARKDOWN, FRAME
from resources.utils import get_class
from resources import config as cfg


def get_equivalence_classes(predictions: np.ndarray) -> np.ndarray:
    num_samples, num_machines, _ = predictions.shape
    ret = np.empty((num_samples, num_machines))

    for i_s in range(num_samples):
        # for each sample, keep the unique predictions
        unique_predictions = []
        for i_m in range(num_machines):
            # get the eqivalence class of the current machine
            current_prediction = predictions[i_s, i_m]
            # this should update unique_predictions if necessary
            current_class = get_class(
                current_prediction, unique_predictions)
            ret[i_s, i_m] = current_class

    return ret


def sort_files(filename):
    machine = filename.split('/')[1]
    return cfg.MACHINE_INFO[machine]['sort'] if machine in cfg.MACHINE_INFO else sys.maxsize


def visualize_uniques(machines: list, distributions: np.ndarray, counts: np.ndarray) -> None:
    # sort by counts, in descending order
    sorted_indices = np.argsort(counts)[::-1]
    total_samples = counts.sum()

    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.align = 'l'
    table.hrules = FRAME
    table.header = False
    table.add_row(["Machine \\ Occurences"] +
                  [f"{c*100/total_samples}%" for c in counts[sorted_indices]])

    for i, machine in enumerate(machines):
        row = [machine]
        for distribution in distributions[sorted_indices]:
            row.append(int(distribution[i]))
        table.add_row(row)

    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", help="The name of the model (e.g. imagenet")
    args = parser.parse_args()

    # find all prediction files and corresponding machines
    files = glob.glob(f"predictions/*/prediction_{args.model_name}_full.npy")
    files.sort(key=sort_files)
    machines = [f.split('/')[1] for f in files]
    machines = list(map(lambda x: cfg.MACHINE_INFO[x]['name'] if x in cfg.MACHINE_INFO else x, machines))

    # load predictions
    predictions = [np.load(f) for f in files]
    # calculate dimensions of combined predictions
    single_prediction_shape = predictions[0].shape
    new_dimensions = (single_prediction_shape[0], len(
        machines), single_prediction_shape[1])

    # combine predictions into single np.ndarray
    combined_predictions = np.empty(new_dimensions)
    for sample_index in range(new_dimensions[0]):
        for machine_index in range(len(machines)):
            combined_predictions[sample_index,
                                 machine_index] = predictions[machine_index][sample_index]

    equivalence_classes = get_equivalence_classes(combined_predictions)
    unique_distributions, counts = np.unique(
        equivalence_classes, return_counts=True, axis=0)

    visualize_uniques(machines, unique_distributions, counts)
