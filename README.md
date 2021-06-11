# iNNformant

This is the accompanying source code repository for our IH&MMSec '21 paper "iNNformant: Boundary Samples as Telltale Watermarks".
In this paper we show that it is possible to identify the hardware used for inference, using only the final output labels.

If you use the code or results, please cite the original paper:

```bibtex
@inproceedings{SKB2021-IH,
	author = {Alexander Schl{\"o}gl and Tobias Kupek and Rainer B{\"o}hme},
	booktitle = {ACM Workshop on Information Hiding and Multimedia Security (IH\&MMSEC)},
	localfile = {SKB2021_IH.pdf},
	title = {{iNNformant}: Boundary Samples as Telltale Watermarks},
	year = {2021}}
```

## CAUTION: Git LFS

We have large `.h5` model saves and many `.npy` predictions in our repository.
To keep the codebase small, we use [git LFS](https://git-lfs.github.com/) to work around this issue.
This also means that when you clone this repository, it will not work unless you have `git lfs` installed on your local machine.

## Development Setup

The code can be used OOTB by simply installing the pip requirements with `pip install -r requirements.txt`, ideally in your environment manager of choice (virtualenv, conda, etc.)

Because we already have a docker container for our cloud experiments, we also have an extended container for development.
At the first sight, this might seem complex, but it helps keep the developmend environments consistent.

### Dockerfile

The `Dockerfile` contains all python dependencies, as well as the [Google-Cloud-SDK](https://cloud.google.com/sdk) which we need to start our experiments in the google cloud.

To build it (from this directory) use the command

```bash
docker build -t innformant-dev --target development .
```

#### Running the Docker

The previously built docker image can be run with (this will fail without the preparation steps below, I just need it for explanation):

```bash
docker run -it \
    -v cloud_config:/credentials \
    --mount type=bind,src=$(pwd),dst=/workspace/iNNformant \
    --mount type=bind,src=$DATASETS,dst=/datasets \
    --mount type=bind,src=$HOME/.ssh,dst=/home/dev/.ssh \
    -e DATASETS=/datasets \
    -e CLOUDSDK_CONFIG=/credentials/gcloud \
    -e SSH_USERNAME=$USER \
    --name innformant \
    -u 1000:1000 \
    -p 1234:1234 \
    innformant-dev
```

As this command is fairly complex, let's go through it one by one.

##### The `cloud_config` Volume

Because the cloud vms use credentials stored on a shared disk, the development container does the same.
Instead of binding the user's gcloud authentication data, we use a docker volume for this.
This procedure is based on the official [cloud-sdk Docker](https://cloud.google.com/sdk/docs/downloads-docker) from Google, but somewhat modified for reusability.

The first step is to create the volume using

```bash
docker volume create cloud_config
```

The newly created volume will be owned by `root`, which can be troublesome if you run docker as root on your host system (this also happens with [Docker Desktop for Windows](https://docs.docker.com/docker-for-windows/install/)).

---

As a short aside, here's the reason why that can be troublesome:
Running docker itself as root will break the automatic mapping from `root` inside the container to your current user outside the container, and any files created as `root` will actually be owned by `root` on the host system.

---

To fix the `root` ownership, we need a bit of a workaround.
Note that you don't have to do this if you use [docker-rootless](https://docs.docker.com/engine/security/rootless/) or [podman](https://podman.io/).

To fix it, mount it in a small image (I use [alpine](https://alpinelinux.org/)), and do a `chown 1000:1000`

```bash
docker run --rm -it -v cloud_config:/credentials alpine chown 1000:1000 /credentials
```

The `1000:1000` are a user id and group id, and will be mapped to the first existing user inside or outside the container.
This should work on most systems, although I haven't tested it on multi-user systems.

##### The Mounts

We `bind` mount three directories, the current working directory (`$(pwd)`), the datasets directory, and the SSH directory.
The working directory is needed to immediately see changes made to the files.
The datasets directory is `bind` mounted save us from copying the datasets into a volume.
And the SSH directory (which includes public and private keyfiles) is user specific, so that should just be mounted from `$HOME`.

Note that the dataset mount requires the `$DATASETS` variable to be set in your shell, containing the absolute path to where you keep your datasets.

##### Environment Variables

The python files in the docker container use environment variables for their configuration.
`$DATASETS` points to the path of the datasets inside the docker.
`$CLOUDSDK_CONFIG` tells `gcloud` where to look for the config, and points into the `cloud_config` volume which is mounted to `/credentials`.
The `$SSH_USERNAME` is again user specific, and required for SSH'ing into the gcloud instances.

When you upload a public key into gcloud metadata, that key will be copied to every newly created instance.
The login does some fancy mapping to usernames, and will grab the username from the public key (can be found at the end).
To not commit any sensitive information, and because it's user specific, this is supplied as an environment variable.

##### User

The user inside the container is set using a uid and gid.
These are both `1000`, which is the first id of both types created.

In our specific container this maps to the `dev` user, which is required for VSCode Devcontainers to work on Windows (more on that below).

##### Others

We also name our container for easier reuse with the `--name` parameter, and publish port 1234, which is sometimes required for testing.
Note that on Linux you may want to use `--network host` to just share your network interface with the development docker, as published ports can't easily be changed for a running container.

The final parameter is simply the name of the image to run, and was specified in the `docker build` command.

### Initializing the Gcloud Credentials

The first time you use a `gcloud` command, it's going to tell you that you're not logged in.
To do so, run `gcloud init`, which will guide you through the rest of the steps.

The authentication and configuration data will automatically be saved in the `cloud_config` volume in a `gcloud` directory, and can be reused later.

### Using VSCode Devcontainer

As a more convenient alternative to running the development docker from the command line, you can use the Devcontainer Plugin for VSCode.
A useful introduction to the concept can be found [here](https://code.visualstudio.com/docs/remote/create-dev-container).
The tl;dr is that the development docker will be run, and VSCode will connect to the docker, with plugins and configuratino values set as specified in the [.devcontainer.json](./.devcontainer.json).
The `devcontainer.json` combines the `docker build` and `docker run` commands into one config file, and I encourage you to take a look inside it, as it's a really neat tool.

If you look at the `mounts` section in the configuration file, you can see that the datasets are mounted from `${localEnv:DATASETS}`.
This is the equivalent of `$DATASETS` in the `docker run` command shown above.
However, because the Devcontainer build does not happen from the bash, the `$DATASETS` variable must be set in your `~/.profile` configuration (as opposed to `~/.bashrc` or whatever shell you use).

To build the Devcontainer and connect to it, install the [Remote - Containers Plugin](vscode:extension/ms-vscode-remote.remote-containers) (if you're on Windows, you're also going to need the WSL remote plugin, and can install both using the [Remote Development Collection](vscode:extension/ms-vscode-remote.vscode-remote-extensionpack)).
Once those are installed, open the project directory in VSCode.
In the bottom right you will be prompted if you want to reopen the current directory in a development container.
Alternatively, you can open the command pallette (`ctrl-shift-p` or `F` and look for "rebuild and reopen in container").

Note that changes to the Dockerfile or the `.devcontainer.json` will require you to rebuild the Devcontainer.
This can be done using the "rebuild and reopen in container" command, or the "rebuild container" command, if you're already connected.

#### Special Windows Setup

Using devcontainer on Windows unfortunately requires some additional preparation steps.
To use the devcontainer on Windows the following tools are necessary:

- Visual Studio Code (+ Extensions `Remote - WSL` and `Remote - Containers`)
- WSL2 (it's important to have your distro with version 2, see [here](https://docs.microsoft.com/en-us/windows/wsl/install-win10#set-your-distribution-version-to-wsl-1-or-wsl-2) how to upgrade)
- Docker Desktop

The `${localEnv:DATASETS}` in the `devcontainer.json` does not work on Windows.
Therefore, you need to clone the project in a WSL distribution of your choice, and set the environment variables there.
Unfortunately, we haven't found a better way to do this yet, but we're open to suggestions.

In summary, the steps for starting development on this project from Windows are the following:

1. Invoke your WSL2 bash
2. Install `git` and `git-lfs`
3. Clone the project
4. Step into the folder
5. Run VSCode `code .`

VSCode magic will deal with the rest, and all remaining steps will be just as for Linux users.

### Check Setup

To check the setup, you can perform a single prediction with `python predict.py mnist`.
Don't worry, this will not be saved into

To confirm the cloud connection, run `gcloud compute instances list`.

If these checks work, your setup is ready to work with the cloud experiment automation via `gcloud.py`.

## Run Experiments

To create attempt creation of the 400 boundary samples we used for our evaluation, simply run the [generate_boundary.py](generate_boundary.py) script.
It takes a model name as parameter (one of "fmnist", "cifar10", or "imagenet"), and will create instances on the Google Compute Platform (GCP) as needed.
Connection creation should also work automatically, if the SSH keys and credentials for GCP are correctly configured.

To generate the samples we discussed in the paper, run the following commands:

```bash
python generate_boundary fmnist
python generate_boundary cifar10
```

The script has two optional flags, that can be passed as options:

- `--remote_only` skips the local phase, and only executes on remote machines
- `--stop_early` ends the generation if more than one partition exists. This is insufficient to identify any microarchitecture.

Neither option was used for the results in the paper.

### Evaluating the experiments

The boundary sample generation process automatically logs data to CSV files.
For our evaluation we used the results in [boundaries/i7-9700/boundary-experiments-7.csv](./boundaries/i7-9700/boundary-experiments-7.csv).
The analysis itself was done in R, using the [1vN.r](analysis/1vN.r) script.
Please note that the script outputs LaTeX macros directly.
If anything is unclear about those macros, feel free to create an issue or reach out.
