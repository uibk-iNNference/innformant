#!/bin/bash


# Alex:
# this is what I use for development
# It bind mounts the root folder to /workspace/innformant
# it also binds the datasets, the path to which is set as an enviromnent variable $DATASETS
# third, it binds the gcloud credentials, which I have in a docker volume called gcloud_config

# it also sets a number of environment variables that our scripts use for the correct folders
# and last but not least, it specifies that we use user 1000:1000 (uid:gid), which maps to dev
# dev is only really needed for the VSCode Devcontainer setup, but it makes things nicer for raw docker as well

# run docker
# SSH_USERNAME is required for the fabric SSH connection from the docker
docker run -it \
    -v gcloud_config:/credentials \
    --mount type=bind,src=$(pwd),dst=/workspace/innformant \
    --mount type=bind,src=$DATASETS,dst=/datasets \
    --mount type=bind,src=$HOME/.ssh,dst=/home/dev/.ssh \
    --env-file development.env \
    --name innformant \
    -u 1000:1000 \
    -p 8888:8888 \
    innformant-dev
