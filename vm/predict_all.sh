#!/bin/bash

set -e

export GOOGLE_APPLICATION_CREDENTIALS="/credentials/bucket.json"

mkdir -p /tmp/predictions

# predict single samples
python predict.py mnist
python predict.py cifar10
python predict.py fmnist
python predict.py imagenet-pretrained
python predict.py imagenet-empty
python predict.py imagenet-trained

# predict 500 samples
python predict.py mnist --full
python predict.py cifar10 --full
python predict.py fmnist --full
python predict.py imagenet-pretrained --full
python predict.py imagenet-empty --full
python predict.py imagenet-empty-quantized-5 --full
python predict.py imagenet-trained --full

python cloud.py predictions upload
