import numpy as np
import tempfile
from resources import cloud
from glob import glob
from os.path import join
import shutil

if __name__ == "__main__":
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
    else:
        print(
            f"No conflicts found, moving new files from {new_base} to {old_base}")
        for path in new_files:
            # TODO: we may need to ensure the subdirectories exist here
            shutil.move(join(new_base, path), join(old_base, path))
        shutil.rmtree(new_base)
