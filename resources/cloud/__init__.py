from . import gcloud_backend as gcloud

from resources import config as cfg

# configure the backends that are used
use_gcloud = True

# All following functions are meant to be easily extendable with additional cloud backends


def get_instance_list(name_filter: str = None) -> list:
    ret = []
    if use_gcloud:
        ret += gcloud.get_instance_list(name_filter)

    return ret

# TODO: cpus should actually be a dictionary with 'backend': ["Intel ...", ...], but I don't want to change too much right now


def start_experiments(cpus: list = cfg.GCLOUD_CPUS) -> None:
    if use_gcloud:
        gcloud.start_experiments(cpus)


def start_servers() -> None:
    if use_gcloud:
        gcloud.start_servers()


def start_test_machine() -> None:
    if use_gcloud:
        gcloud.start_test_machine()


def update_weights() -> None:
    if use_gcloud:
        gcloud.update_weights()

# TODO: add confirmation?


def delete_all(name_filter: str = None) -> None:
    if use_gcloud:
        gcloud.delete_all(name_filter)


def upload_all_predictions(backend: str) -> None:
    if backend == "gcloud":
        gcloud.upload_all_predictions()
    else:
        raise ValueError(f"Unknown cloud backend {backend}")


def download_all_predictions(backend: str, target_dir: str) -> None:
    if backend == "gcloud":
        gcloud.download_all_predictions(target_dir)
    else:
        raise ValueError(f"Unknown cloud backend {backend}")
