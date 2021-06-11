FROM tensorflow/tensorflow:2.3.0 as vm

# Install R for analysis
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y r-base

WORKDIR /innformant

COPY vm/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=1000:1000 resources resources
COPY --chown=1000:1000 *.py ./
COPY --chown=1000:1000 vm/predict_all.sh predict_all.sh

# CMD "bash"
CMD ["bash", "predict_all.sh"]


###### PREDICTION SERVER #######

FROM vm as server
CMD ["python", "server.py"]


###### DEVELOPMENT CONTAINER ########

FROM vm as development

RUN apt-get update -q -y && apt-get install -q -y ssh socat git git-lfs net-tools psmisc

# install gcloud sdk
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - \
    && apt-get update -y && apt-get install google-cloud-sdk -y

# install additional dev dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter for exploration
EXPOSE 8888

# this is required for vscode devcontainers to work
RUN adduser --gecos "" --disabled-password dev
RUN echo 'export PATH=$PATH:~/.local/bin' >> /home/dev/.bashrc

WORKDIR /workspace/innformant

CMD "bash"
