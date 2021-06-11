#!/bin/bash

# exit on error
set -e
# reroute output to file
exec 3>&1 4>&2 >/tmp/startup.log 2>&1

TARGET_USER="forennsic"
TARGET_HOME="/home/$TARGET_USER"
HOSTNAME="$(hostname)"
DELETE_COMMAND="gcloud compute instances delete $(hostname) -q \
--zone $(curl -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone -s | cut -d/ -f4)"
CONTAINER="alxshine/innformant-vm"

# create user forennsic for later use
useradd -G adm,docker -- $TARGET_USER

#create mount points
mkdir -p /mnt/disks/datasets
mkdir -p /mnt/disks/weights
mkdir -p /mnt/disks/credentials

# mount disks
mount -o discard,defaults /dev/disk/by-id/google-datasets /mnt/disks/datasets/
mount -o discard,defaults /dev/disk/by-id/google-weights /mnt/disks/weights/
mount -o discard,defaults /dev/disk/by-id/google-credentials /mnt/disks/credentials/

# readonly gcloud config doesn't work, so copy and fix permissions
cp -r /mnt/disks/credentials/gcloud_config/ $TARGET_HOME/
chown $TARGET_USER:$TARGET_USER $TARGET_HOME/gcloud_config/

# this script is run as root, which can lead to problems as /root is readonly
# create and change to user forennsic

sudo su $TARGET_USER << BASH
# exit on error
set -e
# move into the home directory
cd $TARGET_HOME
touch forennsic_startup_script_was_here

# explicitly pull docker container
docker pull $CONTAINER

# run docker container
docker run \
    --mount type=bind,src=/mnt/disks/weights/,dst=/weights \
    --mount type=bind,src=/mnt/disks/datasets/,dst=/datasets \
    --mount type=bind,src=/mnt/disks/credentials/,dst=/credentials \
    --rm \
    -e HOSTNAME=${HOSTNAME} \
    -e WEIGHTS=/weights \
    -e DATASETS=/datasets \
    -u 1000:1000 \
    $CONTAINER


# delete instance
docker run --rm \
    --mount type=bind,src=/mnt/disks/credentials/,dst=/credentials \
    --mount type=bind,src=$TARGET_HOME/gcloud_config,dst=/config \
    -e CLOUDSDK_CONFIG=/config \
    gcr.io/google.com/cloudsdktool/cloud-sdk \
    $DELETE_COMMAND
BASH

# restore stdout and stderr
exec 1>&3 2>&4