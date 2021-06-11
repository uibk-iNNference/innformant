import os
import tempfile
import getpass
import socket

TMP_FILE = "prediction_tmp.npy"
FILE_TEMPLATE = "prediction_{}.npy"
ARTIFICIAL_KEYS = ['_zero_image', '_half_image', '_one_image', '_horizontal_gradient', '_vertical_gradient',
                   '_diagonal_gradient', '_checkerboard', '_vertical_gradient_1', '_vertical_gradient_2',
                   '_vertical_gradient_3', '_vertical_gradient_4', '_vertical_gradient_5', '_vertical_gradient_6',
                   '_vertical_gradient_14', '_vertical_gradient_15', '_vertical_gradient_28', '_horizontal_gradient_1',
                   '_horizontal_gradient_2',
                   '_horizontal_gradient_3', '_horizontal_gradient_4', '_horizontal_gradient_5',
                   '_horizontal_gradient_6', '_horizontal_gradient_14', '_horizontal_gradient_15',
                   '_horizontal_gradient_28', '_diagonal_gradient_1', '_diagonal_gradient_2', '_diagonal_gradient_3',
                   '_diagonal_gradient_4', '_diagonal_gradient_5', '_diagonal_gradient_6', '_diagonal_gradient_14',
                   '_diagonal_gradient_15', '_diagonal_gradient_28']
GPU_FILE_TEMPLATE = "prediction_{}_gpu.npy"
ENVIRONMENT = "foreNNsic"
PROJECT_DIR = "Projects/forennsic"
PREDICTIONS_DIR = "predictions"
FULL_PREDICTIONS_DIR = f"{PROJECT_DIR}/{PREDICTIONS_DIR}"
USER = "forennsic"

IMAGENET_DATASET_FILENAME = "ILSVRC2012_img_val.tar"
IMAGENET_DOWNLOAD_URL = "http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112" \
                        "/ILSVRC2012_img_val.tar"
NUM_SAMPLES = 500

GCLOUD_SERVICE_ACCOUNT = "forennsic@forennsic.iam.gserviceaccount.com"
GCLOUD_CPUS = [
    "Intel Cascade Lake",
    "Intel Skylake",
    "Intel Broadwell",
    "Intel Haswell",
    "Intel Ivy Bridge",
    "Intel Sandy Bridge",
    "AMD Rome"
]
GCLOUD_ZONE = "europe-west1-b"
GCLOUD_BUCKET = "forennsic"

MACHINE_INFO = {
    'gcloud-intelsandybridge': {'sort': 0, 'name': 'Intel Sandy Bridge'},
    'gcloud-intelivybridge': {'sort': 1, 'name': 'Intel Ivy Bridge'},
    'gcloud-intelhaswell': {'sort': 2, 'name': 'Intel Haswell'},
    'gcloud-intelbroadwell': {'sort': 3, 'name': 'Intel Broadwell'},
    'gcloud-intelskylake': {'sort': 4, 'name': 'Intel Skylake'},
    'gcloud-intelcascadelake': {'sort': 5, 'name': 'Intel Cascade Lake'},
    'gcloud-amdrome': {'sort': 6, 'name': 'AMD Rome'},
}

def get_data_dir():
    try:
        return os.environ['DATASETS']
    except KeyError:
        return 'datasets'


def get_weight_dir():
    try:
        return os.environ['WEIGHTS']
    except KeyError:
        return 'weights'


def get_predictions_dir():
    try:
        return os.environ['PREDICTIONS']
    except KeyError:
        return os.path.join(tempfile.gettempdir(), 'predictions')


def get_ssh_user():
    try:
        return os.environ['SSH_USERNAME']
    except KeyError:
        return getpass.getuser()


def get_prediction_directory():
    return os.path.join(
        get_predictions_dir(),
        get_cleaned_hostname())


def get_cleaned_hostname():
    try:
        return os.environ["HOSTNAME"]
    except KeyError:
        return socket.gethostname().capitalize().replace(' ', '')


DISPLAY_NAMES = {}

SSH_TIMEOUT = 2
