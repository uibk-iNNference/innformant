import argparse
import tempfile
from glob import glob
from os.path import join

from prettytable import PrettyTable
import numpy as np
from resources import cloud


def experiments_start(args):
    cloud.start_experiments()


def experiments_stop(args):
    cloud.delete_all('gcloud')


def experiments_status(args):
    print_instances_table(cloud.get_instance_list('gcloud'))


def servers_start(args):
    cloud.start_servers()


def servers_stop(args):
    cloud.delete_all('server')


def servers_status(args):
    print_instances_table(cloud.get_instance_list('server'))


def list_instances(args):
    print_instances_table(cloud.get_instance_list())


def print_instances_table(instances: list):
    if len(instances) == 0:
        print('no instances running')
        return

    table = PrettyTable()
    table.header = True
    attributes = None

    for instance in instances:
        if attributes is None:
            attributes = [a for a in dir(instance) if not a.startswith('__')]
            table.field_names = attributes
        values = [getattr(instance, a) for a in attributes]
        table.add_row(values)
    print(table)


def test_instances(args):
    cloud.create_test_machine()


def update_weights(args):
    cloud.update_weights()


def upload_predictions(args):
    cloud.upload_all_predictions("gcloud")


def download_predictions(args):
    # first create the directory for new predictions
    new_base = tempfile.mkdtemp(prefix="innformant", suffix="predictions_new")
    new_base_length = len(new_base.split('/'))
    print("Downloading predictions...")
    cloud.download_all_predictions("gcloud", new_base)

    old_base = "predictions"
    new_paths = glob(join(new_base, '*', '*.npy'))

    new_files = []
    conflicting_files = []

    for path in new_paths:
        common_path = path.split('/')[new_base_length:]

        try:
            old_prediction = np.load(join(old_base, *common_path))
        except FileNotFoundError:
            new_files.append('/'.join(common_path))
            continue

        new_prediction = np.load(path)
        if not np.all(old_prediction == new_prediction):
            conflicting_files.append('/'.join(common_path))

    if len(new_files) > 0:
        print(f"Found {len(new_files)} new files:")
        for path in new_files:
            print(path)

        print("\n")

    if len(conflicting_files) > 0:
        print(f"Found {len(conficting_files)} conflicting files:")
        for path in conficting_files:
            print(path)

        print("\n")

    if len(conflicting_files) == 0 and len(new_files) > 0:
        print(
            f"No conflicts found, moving new files from {new_base} to {old_base}")
        for path in new_files:
            # TODO: we may need to ensure the subdirectories exist here
            shutil.move(join(new_base, path), join(old_base, path))
        shutil.rmtree(new_base)
    else:
        print("No changes detected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Control gcloud servers and experiments.')
    subparsers = parser.add_subparsers(title='commands', dest='command')

    subparsers.add_parser(
        'list', help='Get all existing vm instances and exit').set_defaults(func=list_instances)
    subparsers.add_parser(
        'test', help='Create a test machine and exit').set_defaults(func=test_instances)
    subparsers.add_parser(
        'update', help='Update weights on gcloud disk').set_defaults(func=update_weights)
    #  subparsers.add_parser('upload', help='Upload all predictions to gcloud bucket').set_defaults(func=upload_predictions)
    subparsers.required = True

    parser_pred = subparsers.add_parser(
        'predictions', help='Utility functions for handling predictions')
    subparser_pred = parser_pred.add_subparsers(
        title='predictions commands', dest='predictions command')
    subparser_pred.add_parser(
        'upload', help='Upload predictions to storage').set_defaults(func=upload_predictions)
    subparser_pred.add_parser('download', help='Download and check new predictions').set_defaults(
        func=download_predictions)
    subparser_pred.required = True

    parser_exp = subparsers.add_parser(
        'experiments', help='Control a series of experiments')
    subparser_exp = parser_exp.add_subparsers(
        title='experiments commands', dest='experiments command')
    subparser_exp.add_parser(
        'start', help='Start a series of experiments').set_defaults(func=experiments_start)
    subparser_exp.add_parser('stop', help='Stop all experiments').set_defaults(
        func=experiments_stop)
    subparser_exp.add_parser('status', help='Get status of running experiments').set_defaults(
        func=experiments_status)
    subparser_exp.required = True

    parser_serv = subparsers.add_parser(
        'servers', help='Control prediction servers')
    subparser_serv = parser_serv.add_subparsers(
        title='servers commands', dest='servers command')
    subparser_serv.add_parser(
        'start', help='Start servers').set_defaults(func=servers_start)
    subparser_serv.add_parser(
        'stop', help='Stop all servers').set_defaults(func=servers_stop)
    subparser_serv.add_parser(
        'status', help='Get status of running servers').set_defaults(func=servers_status)
    subparser_serv.required = True

    args = parser.parse_args()
    args.func(args)
