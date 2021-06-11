#!/bin/bash

python compare_results.py --model_types mnist fmnist cifar10 --hosts brokkr seclab-serv cluster gcloud-broadwell gcloud-sandy-bridge gcloud-skylake amazon-broadwell amazon-skylake amazon-xeon --tex

printf "\n\n\n"