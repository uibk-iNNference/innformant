import numpy as np
import requests
import sys
import json

from resources.networking import PredictionRequest, PredictionResponse, get_remote_prediction
from resources import data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 1234

    print("Building request")
    sample, label = data.get_single_mnist_test_sample(include_label=True)
    prediction_request = PredictionRequest("mnist", sample, label)

    url = "http://host.docker.internal:1234"
    prediction_response = get_remote_prediction(url, prediction_request)
    print(prediction_response.prediction)
