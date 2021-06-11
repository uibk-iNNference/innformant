import numpy as np

from resources import data, networking, utils

def quantize_sample(sample):
    return (sample*255).astype(np.int)

def test_boundary_sample(sessions, model_type, sample, label):
    quantized = quantize_sample(sample)
    normalized, _ = data.normalize_img(1, quantized, 0)
    normalized = normalized.numpy()

    prediction_request = networking.PredictionRequest(
            model_type,
            normalized,
            label)

    labels = {}
    for session in sessions:
        print(f"Getting prediction from {session.name}")
        url = session.build_url()
        prediction_response = networking.get_remote_prediction(url, prediction_request)
        labels[session.name] = np.argmax(prediction_response.prediction)

    for key,value in labels.items():
        print(f"{key}: {value}")
    values = list(labels.values())
    if any(values != values[0]):
        print("SUCCESS :)")
    else:
        print("ERROR :/")

if __name__ == "__main__":
    model_type = "fmnist"
    sessions = networking.open_server_sessions()

    for index in range(100):
        try:
            path = f"boundaries/inteli7-1065/{model_type}_{index}.npy"
            sample = np.load(path)
            quantized = quantize_sample(sample)

            original, label = data.get_single_fmnist_test_sample(index, include_label=True)
            original_quantized = quantize_sample(original)

            diff = np.abs(original_quantized - quantized)
            print(f"max diff: {np.max(diff)}, min diff: {np.min(diff)}, mean diff: {np.mean(diff)}")
            print(f"psnr on qunatized images: {utils.calculate_psnr(original_quantized, quantized, 255)}")

            print("Testing with servers...")
            test_boundary_sample(sessions, model_type, sample, label)
        except IOError:
            pass

