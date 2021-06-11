import os
import shutil
import sys
import time
from os.path import join
from subprocess import CalledProcessError, PIPE, STDOUT, run
from google.cloud import storage

from resources import config as cfg

cpus = cfg.GCLOUD_CPUS
zone = cfg.GCLOUD_ZONE

# default machine type will be n1-stardard-2
special_machine_types = {
    "AMD Rome": "n2d-standard-2",
    "Intel Cascade Lake": "n2-standard-2"
}


def _clean_cpu_name(cpu: str):
    return cpu.lower().replace('-', '').replace(' ', '')


def get_experiment_name(cpu: str):
    return "gcloud-" + _clean_cpu_name(cpu)


def get_server_name(cpu: str):
    return "server-" + _clean_cpu_name(cpu)


class Instance(object):
    def __init__(self, name, zone, machine_type, ip, status) -> None:
        super().__init__()
        self.name = name
        self.zone = zone
        self.machine_type = machine_type
        self.ip = ip
        self.status = status


def get_instance_list(name_filter: str = None):
    response = run(
        ["gcloud", "compute", "instances", "list"], stdout=PIPE, stderr=STDOUT)
    lines = [line.decode("utf-8") for line in response.stdout.splitlines()]

    ret = []
    for i, line in enumerate(lines):
        if i == 0:
            continue  # first line is header
        parts = line.split()
        if len(parts) == 6:
            ret.append(Instance(parts[0], parts[1],
                                parts[2], parts[4], parts[5]))
        else:
            ret.append(Instance(parts[0], parts[1], parts[2], '', parts[4]))

    if name_filter is not None:
        return list(filter(lambda i: name_filter in i.name, ret))

    return ret


def start_experiments(cpus: list = cfg.GCLOUD_CPUS, zone: str = zone) -> None:
    print(f"Starting {len(cpus)} experiments on zone {zone}")
    startup_path = join('.', 'vm', 'startup.sh')
    for cpu in cpus:
        instance_name = get_experiment_name(cpu)
        try:
            machine_type = special_machine_types[cpu]
        except KeyError:
            machine_type = "n1-standard-2"

        print(f"Creating instance with {cpu} processor")

        finished_process = run([
            "gcloud", "compute", "instances", "create",
            instance_name,
            "--min-cpu-platform", cpu,
            "--machine-type", machine_type,
            "--maintenance-policy", "TERMINATE",
            "--service-account", cfg.GCLOUD_SERVICE_ACCOUNT,
            "--zone", zone,
            "--image-family", "cos-stable",
            "--image-project", "cos-cloud",
            "--boot-disk-size", "20GB",
            "--disk", "name=weights,mode=ro,device-name=weights",
            "--disk", "name=datasets,mode=ro,device-name=datasets",
            "--disk", "name=credentials,mode=ro,device-name=credentials",
            "--scopes", "default,cloud-platform,compute-rw",
            "--metadata-from-file", f"startup-script={startup_path}"
        ])

        try:
            finished_process.check_returncode()
        except CalledProcessError:
            print(
                f"ERROR occured during experiment start, aborting:\n{finished_process.stderr}", file=sys.stderr)
            return


def start_servers(cpus: list = cpus, zone: str = zone):
    print(f"Starting {len(cpus)} servers on zone {zone}")
    for cpu in cpus:
        instance_name = get_server_name(cpu)
        try:
            machine_type = special_machine_types[cpu]
        except KeyError:
            machine_type = "n1-standard-2"

        print(f"Creating instance with {cpu} processor")

        finished_process = run([
            "gcloud", "compute", "instances", "create-with-container",
            instance_name,
            "--min-cpu-platform", cpu,
            "--machine-type", machine_type,
            "--service-account", cfg.GCLOUD_SERVICE_ACCOUNT,
            "--zone", zone,
            "--disk", "name=weights,mode=ro,device-name=weights",
            "--disk", "name=datasets,mode=ro,device-name=datasets",
            "--disk", "name=credentials,mode=ro,device-name=credentials",
            "--container-mount-disk", "name=credentials,mount-path=/credentials,mode=ro",
            "--container-mount-disk", "name=weights,mount-path=/weights,mode=ro",
            "--container-mount-disk", "name=datasets,mount-path=/datasets,mode=ro",
            "--container-env-file", "vm/environment.env",
            "--container-image", "docker.io/alxshine/innformant-server"
        ])

        try:
            finished_process.check_returncode()
        except CalledProcessError:
            print(
                f"ERROR occured during experiment start, aborting:\n{finished_process.stderr}", file=sys.stderr)


def create_test_machine(zone: str = zone):
    print(f"Starting test machine on zone {zone}")
    instance_name = 'innformant-test'
    run([
        "gcloud", "compute", "instances", "create", instance_name,
        "--maintenance-policy", "TERMINATE",
        "--service-account", cfg.GCLOUD_SERVICE_ACCOUNT,
        "--zone", zone,
        "--image-family", "cos-stable",
        "--image-project", "cos-cloud",
        "--boot-disk-size", "20GB",
        "--disk", "name=weights,mode=ro,device-name=weights",
        "--disk", "name=datasets,mode=ro,device-name=datasets",
        "--disk", "name=credentials,mode=ro,device-name=credentials",
        "--scopes", "default,cloud-platform,compute-rw"])


def update_weights(zone: str = zone):
    print(f"Starting maintenance machine on zone {zone}")
    instance_name = 'innformant-maintenance'
    run([
        "gcloud", "compute", "instances", "create", instance_name,
        "--machine-type", "n1-standard-1",
        "--maintenance-policy", "TERMINATE",
        "--service-account", cfg.GCLOUD_SERVICE_ACCOUNT,
        "--zone", zone,
        "--disk", "name=weights,mode=rw,device-name=weights,auto-delete=no",
        "--disk", "name=datasets,mode=rw,device-name=datasets,auto-delete=no"])

    instances = get_instance_list()
    # test if any instance with our maintenance name exists
    while not any([instance_name in instance.name for instance in instances]):
        time.sleep(3)
        instances = get_instance_list()

    print("Maintenance instance created")

    print("Mounting weights disk to VM")
    # the first ssh can fail because the machine is not set up yet
    return_code = 1
    while return_code:
        return_code = run(["gcloud", "compute", "ssh",
                           instance_name,
                           "--zone", zone,
                           "--command", "ls"]).returncode

    run(["gcloud", "compute", "ssh",
         instance_name,
         "--zone", zone,
         "--command", "sudo mkdir -p /mnt/disks/weights"], check=True)
    run(["gcloud", "compute", "ssh",
         instance_name,
         "--zone", zone,
         "--command",
         "sudo mount /dev/disk/by-id/google-weights /mnt/disks/weights"],
        check=True)

    print("Transfering weights")
    run(["gcloud", "compute", "scp",
         "--compress",
         "--zone", zone,
         "--recurse",
         "weights",
         f"root@{instance_name}:/mnt/disks/"
         ], check=True)

    print("Deleting instance")
    run(["gcloud", "compute", "instances", "delete",
         "-q",
         instance_name,
         "--zone", zone], check=True)

    print("Update finished :)")


def delete_all(name_filter: str = None) -> None:
    print("Retrieving running instances...")
    instances = get_instance_list()
    if name_filter is not None:
        instances = filter(lambda i: name_filter in i.name, instances)
    instances = list(map(lambda i: i.name, instances))

    print(f"Deleting the following instances: {instances}")
    if (len(instances) is 0):
        return

    run(["gcloud", "compute", "instances", "delete",
         "--zone", zone,
         "-q"] + instances)


def upload_all_predictions():
    predictions_path = cfg.get_prediction_directory()
    prediction_files = [f for f in os.listdir(
        predictions_path) if os.path.isfile(os.path.join(predictions_path, f))]

    for file in prediction_files:
        upload_prediction(os.path.join(predictions_path, file),
                          f'predictions/{cfg.get_cleaned_hostname()}/{file}')


def upload_prediction(source_file_name, destination_file_name):
    upload_blob(cfg.GCLOUD_BUCKET, source_file_name, destination_file_name)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def download_all_predictions(target_dir: str) -> None:
    run(["gsutil", "-m",
         "cp", "-r",
         "gs://forennsic/predictions/*",
         target_dir])
