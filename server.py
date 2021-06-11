import json

import numpy as np
import tensorflow as tf
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response

from resources import utils, predictions
from resources.networking import PredictionRequest, PredictionResponse

@Request.application
def handle_request(request: Request) -> Response:
    # get the prediction request from request
    payload = request.data
    json_payload = json.loads(payload)
    prediction_request = PredictionRequest.from_json(json_payload)
    prediction_response = predict(prediction_request)
    response_payload = prediction_response.to_json()
    return Response(json.dumps(response_payload), mimetype='application/json')

def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        if request.label is not None:
            # if a label was supplied, then assume we need to return a gradient as well
            prediction, gradient = predictions.predict_with_gradient(
                request.model_type, request.sample, request.label)
            return PredictionResponse(
                    request.model_type,
                    request.sample,
                    request.label,
                    prediction,
                    gradient,
                    None)
        else:
            # if no label is supplied, do a standard prediction
            prediction = predictions.predict(
                request.model_type, request.sample)
            return Response(request.model_type, request.sample, None, prediction, None, None)
    except tf.errors.InvalidArgumentError as e:
        return Response(request.model_type, request.sample, None, f"ERROR during prediction: {e.message}")

if __name__ == "__main__":
    run_simple('0.0.0.0', 1234, handle_request)
