import argparse
import os
import sys
from functools import partial

import numpy as np
import pandas as pd

from typing import List, Union

from joblib import Parallel, delayed

from resources import data, predictions, utils, config as cfg
from resources.networking import Session, PredictionRequest, open_server_sessions, get_remote_prediction


class PerformanceStat(object):
    def __init__(self, num_steps, psnr):
        self.num_steps = num_steps
        self.psnr = psnr


class InstanceStat(object):
    def __init__(self, hostname, label, conf1, conf2):
        self.hostname = hostname
        self.label = label
        self.conf1 = conf1
        self.conf2 = conf2


class LocalStat(object):
    def __init__(self, performance_stat: PerformanceStat, instance_stat: InstanceStat):
        self.performance_stat = performance_stat
        self.instance_stat = instance_stat

    def to_flat_dict(self):
        return {
            'local_steps': self.performance_stat.num_steps,
            'local_psnr': self.performance_stat.psnr,
            'local_label': self.instance_stat.label,
            'local_conf1': self.instance_stat.conf1,
            'local_conf2': self.instance_stat.conf2
        }


class RemoteStats(object):
    def __init__(self, performance_stat: PerformanceStat, instance_stats: List[InstanceStat]):
        self.performance_stat = performance_stat
        self.instance_stats = instance_stats

    def to_flat_dict(self):
        ret = {
            'remote_steps': self.performance_stat.num_steps,
            'remote_psnr': self.performance_stat.psnr
        }

        for i, stat in enumerate(self.instance_stats):
            ret[f"s{i}_hostname"] = stat.hostname
            ret[f"s{i}_label"] = stat.label
            ret[f"s{i}_conf1"] = stat.conf1
            ret[f"s{i}_conf2"] = stat.conf2

        return ret


class Measurements(object):
    def __init__(self, identified_MA: str, model_type: str, sample_index: int, total_psnr: float, local_stat: LocalStat, remote_stats: RemoteStats):
        self.identified_MA = identified_MA
        self.model_type = model_type
        self.sample_index = sample_index
        self.total_psnr = total_psnr

        self.local_stat = local_stat
        self.remote_stats = remote_stats

    def to_flat_dict(self):
        result = {
            'identified_MA': self.identified_MA,
            'model_type': self.model_type,
            'sample_index': self.sample_index,
            'total_psnr': self.total_psnr,
        }
        if self.local_stat is not None:
            result.update(**self.local_stat.to_flat_dict())
        if self.remote_stats is not None:
            result.update(**self.remote_stats.to_flat_dict())
        return result


class ServerPrediction(object):
    def __init__(self, session_name, prediction, label, conf1, conf2, gradient):
        self.session_name = session_name
        self.prediction = prediction
        self.label = label
        self.conf1 = conf1
        self.conf2 = conf2
        self.gradient = gradient

    def get_conf_diff(self) -> float:
        return np.abs(self.conf1-self.conf2)


class ParallelSession(object):
    def __init__(self, name, url):
        self.name = name
        self.url = url


def local_boundary_step(model_type, sample, real_label, target_confidence=1e-5, max_iterations=2000) -> (np.ndarray, LocalStat):
    """
    An iterative FGSM which scales alpha with the distance to the decision boundary.
    Will bring the sample very close to the boundary.
    """
    def _build_local_stat():
        """ my little helper so I don't have to repeat this code """
        return LocalStat(
            PerformanceStat(
                iterations,
                utils.calculate_psnr(sample, boundary)
            ),
            InstanceStat(
                cfg.get_cleaned_hostname(),
                label,
                confidence,
                confidence_second
            )
        )

    boundary = sample.copy()
    alpha = 0.0001  # main alpha value
    alpha_scaling = 1  # scaling factor

    iterations = 0
    initial_label = None
    last_label = None
    last_confidence = None
    correctness_value = None

    while iterations < max_iterations:
        prediction, gradient = predictions.predict_with_gradient(
            model_type, sample, real_label)
        prediction = prediction[0]
        label = np.argmax(prediction)
        confidence = np.max(prediction)
        confidence_second = np.sort(prediction)[-2]
        confidence_diff = np.abs(confidence - confidence_second)

        if iterations == 0:
            initial_label = label
            last_label = label
            last_confidence = confidence
            correctness_value = 1 if initial_label == np.argmax(
                real_label) else -1

        print(f"{iterations}\tLabel: {label}\tClass diff: {confidence_diff}")

        if confidence == 1.0:  # Issue https://git.uibk.ac.at/c7031199/innformant/issues/6
            print(f"\nFailed - cannot apply FGSM with zero gradient")
            return None, _build_local_stat()

        if confidence_diff > target_confidence:
            label_flip = True if label != last_label else False

            # adjust scaling factor
            # make sure to not get stuck in local optima
            alpha_scaling *= 2 if iterations < 500 else 1.5
            if label_flip or (confidence != last_confidence):
                alpha_scaling = confidence_diff

            # add gradient to sample
            signed_grad = np.sign(gradient)
            correctness_sign = correctness_value if label == initial_label else -correctness_value
            sample = (correctness_sign * signed_grad *
                      alpha * alpha_scaling) + sample

            iterations += 1
            last_label = label
            last_confidence = confidence
        else:
            print("\nSuccess - reached target difference")
            print(f"Sorted confidences:\n{np.sort(prediction)}\n")

            return sample, _build_local_stat()

    print(
        f"\nFailed - not close enough to boundary after {max_iterations} iterations")
    return None, _build_local_stat()


def get_server_prediction(session: ParallelSession, model_type, sample, real_label) -> ServerPrediction:
    response = get_remote_prediction(
        session.url, PredictionRequest(model_type, sample, real_label))

    prediction = response.prediction[0]
    label = np.argmax(prediction)
    confidence = np.max(prediction)
    confidence_second = np.sort(prediction)[-2]
    gradient = response.gradient

    return ServerPrediction(session.name, prediction, label, confidence, confidence_second, gradient)


def server_boundary_step(sessions: List[Session], model_type: str, sample: np.ndarray, real_label: np.ndarray, stop_early: bool, max_iterations=500) -> (np.ndarray, RemoteStats):
    """
    Runs the iterative FGSM with two machine specific gradients.
    Generates a sample with different labels on two machines.
    """
    def _build_remote_stats():
        """ my little helper so I don't have to repeat this code """
        return RemoteStats(
            PerformanceStat(
                iterations,
                utils.calculate_psnr(sample, boundary)
            ),
            [InstanceStat(result.session_name, result.label,
                          result.conf1, result.conf2) for result in results]
        )

    def _identify_MA(partition: List[ServerPrediction]):
        if len(partition) == 1 and 'amd' in partition[0].session_name:
            return 'amd'
        elif len(partition) == 2 and all(['lake' in r.session_name for r in partition]):
            return 'intel-xlake'
        elif len(partition) == 2 and all(['bridge' in r.session_name for r in partition]):
            return 'intel-xbridge'
        elif len(partition) == 2 and all(['well' in r.session_name for r in partition]):
            return 'intel-xwell'
        else:
            return None

    print("Starting remote phase...")
    boundary = sample.copy()

    alpha = 0.0002  # main alpha value
    alpha_scaling = 1  # scaling factor

    iterations = 0

    initial_label = None
    last_label = None
    last_confidence = None
    correctness_value = None
    parallel_sessions = [ParallelSession(
        session.name, session.build_url()) for session in sessions]

    while iterations < max_iterations:
        # get predictions from servers
        partial_get_server_prediction = partial(
            get_server_prediction, model_type=model_type, sample=boundary, real_label=real_label)
        results = Parallel(n_jobs=len(parallel_sessions), prefer="threads")(delayed(
            partial_get_server_prediction)(session) for session in parallel_sessions)

        # partition by their label
        partitions = {}
        for prediction in results:
            try:
                partitions[prediction.label].append(prediction)
            except KeyError:
                partitions[prediction.label] = [prediction]

        # select smallest partition
        smallest_partition = sorted(partitions.values(), key=len)[0]

        # if the partition size is 1, we are done
        if (stop_early and len(partitions) > 1):
            print("\nSUCCESS: stopping early because a label flipped")
            print(
                f"labels: {[prediction.label for prediction in predictions.values()]}")
            identified = smallest_partition[0].session_name
            return boundary, identified, _build_remote_stats()

        identified = _identify_MA(smallest_partition)
        if identified is not None:
            print(f"SUCCESS: found a boundary sample identifying {identified}")
            return boundary, identified, _build_remote_stats()

        # sort partition by their confidence diff
        sorted_partition = sorted(
            smallest_partition, key=lambda p: p.get_conf_diff())
        if len(sorted_partition) == len(sessions):
            # if we are fully outside all boundaries, target the closest boundary
            target = sorted_partition[0]
        else:
            # select the second furthest boundary as target
            target = sorted_partition[-2]

        signed_grad = np.sign(target.gradient)

        if iterations == 0:
            initial_label = target.label
            last_label = target.label
            correctness_value = 1 if initial_label == np.argmax(
                real_label) else -1
            last_confidence = 0

        if target.conf1 == 1.0:  # Issue https://git.uibk.ac.at/c7031199/innformant/issues/6
            print(f"\nFailed - cannot apply FGSM with zero gradient")
            return None, "failure", _build_remote_stats()

        print(
            f"{iterations}\tLabel: {target.label}\tSize of smallest partition: {len(smallest_partition)}, target dconf: {target.get_conf_diff()}")

        label_flip = True if target.label != last_label else False

        # adjust scaling factor
        alpha_scaling *= 1.5 if iterations < 25 else 1.25
        if label_flip or (target.conf1 != last_confidence):
            alpha_scaling = target.get_conf_diff()
            if alpha_scaling == 0:
                alpha_scaling = 1e-7

        # add gradient to boundary
        correctness_sign = correctness_value if target.label == initial_label else -correctness_value
        boundary = (correctness_sign * signed_grad *
                    alpha * alpha_scaling) + boundary

        iterations += 1
        last_label = target.label
        last_confidence = target.conf1

    print(f"\nFailed - no boundary sample after {max_iterations} iterations")
    return None, "failure", _build_remote_stats()


def log_measurements(measurements: Measurements):
    if measurements.remote_stats is None:
        print("Did not do any remote work, samples is \"broken\"")
        return
    num_servers = len(measurements.remote_stats.instance_stats)
    logfile_path = os.path.join('boundaries', cfg.get_cleaned_hostname(
    ), f'boundary-experiments-{num_servers}.csv')
    value_dict = measurements.to_flat_dict()

    try:
        dataframe = pd.read_csv(logfile_path)
        print(f"Updating measurements at {logfile_path}")
    except IOError:
        dataframe = pd.DataFrame(columns=value_dict.keys())
        print(f"Writing measurements to {logfile_path}")

    dataframe = dataframe.append(value_dict, ignore_index=True)
    dataframe.to_csv(logfile_path, header=True, index=False)


def generate_boundary(sample, label, model_type, sessions, stop_early):
    def _build_measurements(identified_MA: str, sample: np.ndarray, boundary: Union[np.ndarray, None], local_stats: Union[LocalStat, None], remote_stats: Union[RemoteStats, None]):
        if boundary is not None and np.any(boundary != sample):
            total_psnr = utils.calculate_psnr(sample, boundary)
        else:
            total_psnr = 0
        return Measurements(identified_MA, model_type, sample_index, total_psnr, local_stats, remote_stats)

    try:
        boundary = sample
        local_stats = None
        if not remote_only:
            boundary, local_stats = local_boundary_step(
                model_type, boundary, label)
            if boundary is None:
                log_measurements(_build_measurements(
                    "failure", sample, boundary, local_stats, None))
                return

        boundary, identified_MA, remote_stats = server_boundary_step(
            sessions, model_type, boundary, label, stop_early)

        log_measurements(_build_measurements(
            identified_MA, sample, boundary, local_stats, remote_stats))
        if boundary is not None:
            save_path = os.path.join('boundaries', cfg.get_cleaned_hostname(
            ), f'{model_type}_{sample_index}_n{len(sessions)}.npy')
            print(f'Saving boundary sample at {save_path}')
            np.save(save_path, boundary)
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", help="The name of the model (e.g. imagenet)")
    parser.add_argument("--remote_only", dest='remote_only', action='store_true',
                        default=False, help="Skip local preparation step")
    parser.add_argument("--stop_early", dest='stop_early', action='store_true',
                        default=False, help="Stop when at least one label is different")
    args = parser.parse_args()

    model_type = args.model_name
    remote_only = args.remote_only
    stop_early = args.stop_early

    sessions = open_server_sessions()

    # for our paper we attempted the first 400 samples in the dataset
    model_type = model_type.lower()
    for sample_index in range(1, 400):
        if model_type == 'mnist':
            sample, label = data.get_single_mnist_test_sample(
                index=sample_index, include_label=True)
        elif model_type == 'cifar10':
            sample, label = data.get_single_cifar10_test_sample(
                index=sample_index, include_label=True)
        elif model_type == 'fmnist':
            sample, label = data.get_single_fmnist_test_sample(
                index=sample_index, include_label=True)
        elif model_type == 'imagenet':
            sample, label = data.get_single_imagenet_test_sample(
                index=sample_index, include_label=True)
        else:
            raise RuntimeError(f"Unknown model type {model_type}")

        generate_boundary(sample, label, model_type, sessions, stop_early)
