import numpy as np
import base64
import json
import requests
import subprocess
import os
import time

from typing import List

from resources import cloud, config as cfg


def _encode_np_array(array: np.ndarray) -> dict:
    if array is None:
        return None

    array_bytes = array.tobytes()
    encoded_bytes = base64.standard_b64encode(array_bytes)
    decoded_bytes = encoded_bytes.decode("UTF-8")
    ret = {
        'bytes': decoded_bytes,
        'dtype': str(array.dtype),
        'shape': array.shape
    }
    return ret


def _decode_np_array(encoded_array: dict) -> np.ndarray:
    if encoded_array is None:
        return None
    byte_string = encoded_array['bytes']
    dtype = encoded_array['dtype']
    shape = encoded_array['shape']

    array_string = byte_string.encode("UTF-8")
    array_bytes = base64.standard_b64decode(array_string)
    array = np.frombuffer(array_bytes, dtype=dtype)
    return array.reshape(shape)


class PredictionRequest(object):
    """ Used to request a prediction from the server.
    If the label is not None, then it's assumed a gradient is also desired.
    """

    def __init__(self, model_type: str, sample: np.ndarray, label: np.ndarray):
        self.model_type = model_type
        self.sample = sample
        self.label = label

    def to_json(self) -> dict:
        sample = _encode_np_array(self.sample)

        return {
            "model_type": self.model_type,
            "sample": _encode_np_array(self.sample),
            "label": _encode_np_array(self.label)
        }

    @staticmethod
    def from_json(data: dict):
        model_type = data['model_type']

        sample = _decode_np_array(data['sample'])
        label = _decode_np_array(data['label'])
        return PredictionRequest(model_type, sample, label)

    def __str__(self):
        return f"PredictionRequest: model_type={self.model_type}, sample.shape={self.sample.shape}"


class PredictionResponse(object):
    def __init__(self, model_type: str, sample: np.ndarray, label: np.ndarray, prediction: np.ndarray, gradient: np.ndarray, message: str):
        self.model_type = model_type
        self.sample = sample
        self.label = label
        self.prediction = prediction
        self.gradient = gradient
        self.message = message

    def to_json(self) -> dict:
        return {
            "model_type": self.model_type,
            "sample": _encode_np_array(self.sample),
            "label": _encode_np_array(self.label),
            "prediction": _encode_np_array(self.prediction),
            "gradient": _encode_np_array(self.gradient),
            "message": self.message
        }

    @staticmethod
    def from_json(data: dict):
        model_type = data['model_type']
        message = data['message']

        sample = _decode_np_array(data['sample'])
        label = _decode_np_array(data['label'])

        prediction = _decode_np_array(data['prediction'])
        gradient = _decode_np_array(data['gradient'])

        return PredictionResponse(model_type, sample, label, prediction, gradient, message)

    def __str__(self):
        return f"Response: model_type={self.model_type}, sample.shape={self.sample.shape}, prediction.shape={self.prediction.shape}, message={self.message}"


class Session(object):
    def __init__(self, name: str, ip: str, local_port: int, tunnel: subprocess.Popen):
        self.name = name
        self.ip = ip
        self.local_port = local_port
        self.tunnel = tunnel

    def build_url(self):
        return f"http://localhost:{self.local_port}"

def get_remote_prediction(url: str, prediction_request: PredictionRequest) -> PredictionResponse:
    payload = prediction_request.to_json()
    response = requests.post(url, json=payload)
    prediction_response = PredictionResponse.from_json(response.json())
    return prediction_response


def open_server_sessions() -> List[Session]:
    print("Testing if servers are already running...")
    servers = cloud.get_instance_list("server")

    # test with just 2 servers
    #  servers = servers[0:2]

    if len(servers) == 0:
        # start servers
        print("None found, starting...")
        cloud.start_servers()
        servers = cloud.get_instance_list("server")
        # give the servers some time to get ready
        print("Waiting for servers to set up...")
        time.sleep(60)

    # build SSH tunnel to each server
    print("Opening SSH tunnel...")
    local_port = 1100
    sessions = []
    known_hosts_path = os.path.expanduser('~/.ssh/known_hosts')

    # map from instance_name to local port
    for server in servers:
        # remove key from known hosts
        print(
            f"Removing server key for {server.name}: {server.ip}, from known_hosts {known_hosts_path}")
        subprocess.run(["ssh-keygen",
                        "-f", known_hosts_path,
                        "-R", server.ip])

        cmd = ['ssh',
               '-4',  # this forces IPv4, which docker requires
               '-L', f'{local_port}:localhost:1234',
               '-N',  # ! this is the important bit, it makes the SSH command not do anything
               '-l', cfg.get_ssh_user(),
               # this bypasses the confirmation dialogue before we connect
               '-o', 'StrictHostKeyChecking=no',
               server.ip]

        print(f"Opening tunnel to IP {server.ip} on port {local_port}")
        tunnel = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # this should help ensure the tunnel is established, because Popen is non-blocking
        time.sleep(2)
        print(F"Connecting to localhost on port {local_port}")
        sessions.append(Session(server.name, server.ip, local_port, tunnel))

        local_port += 1

    return sessions
