"""Generic SM inference.py for all TensorFlow inference tasks.

Read metadata and dispatch to task-specifc pipelines
https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#deploying-directly
-from-model-artifacts
https://github.com/aws/sagemaker-tensorflow-serving-container
https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving
.html
https://cloud.google.com/blog/topics/developers-practitioners/add-preprocessing-functions-tensorflow-models-and-deploy-vertex-ai
https://towardsdatascience.com/serving-image-based-deep-learning-models-with-tensorflow-servings-restful-api-d365c16a7dc4
"""

import base64
import json

import requests


LABELS_INFO = "/opt/ml/model/labels_info.json"
PROBABILITIES = "probabilities"
LABELS = "labels"
PREDICTED_LABEL = "predicted_label"
PREDICTIONS = "predictions"
VERBOSE_EXTENSION = ";verbose"


with open(LABELS_INFO, "r") as txtFile:
    labels = json.loads(txtFile.read())[LABELS]


def handler(data, context):
    """Handle request.

    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """

    processed_input = _process_input(data, context)
    response = requests.post(context.rest_uri, data=processed_input)
    return _process_output(response, context)


def _process_input(data, context):
    """Encode input data to base64.

    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        string: data for input to model.
    Raises:
        ValueError: if context.request_content_type is not the expected REQUEST_CONTENT_TYPE
    """
    if context.request_content_type == "application/x-image":
        encoded_input_string = base64.b64encode(data.read())
        input_string = encoded_input_string.decode("utf-8")
        img = json.dumps({"instances": [{"b64": input_string}]})
        # for multiple images: {"instances": [{"b64": "iVBORw"}, {"b64": "pT4rmN"}, {"b64": "w0KGg2"}]}
        return str(img)
    raise ValueError(f"unsupported content type {context.request_content_type or 'unknown'}")


def _process_output(data, context):
    if data.status_code != 200:
        raise ValueError(data.content.decode("utf-8"))
    response_content_type = context.accept_header
    prediction = data.content
    prediction = json.loads(prediction)[PREDICTIONS][0]
    output = {PROBABILITIES: prediction}

    if response_content_type.endswith(VERBOSE_EXTENSION):
        output[LABELS] = labels
        predicted_label_idx = prediction.index(max(prediction))
        output[PREDICTED_LABEL] = labels[predicted_label_idx]
        response_content_type = response_content_type.strip(VERBOSE_EXTENSION)

    output = json.dumps(output)
    return output, response_content_type
