import boto3
import logging
from botocore.exceptions import BotoCoreError, ClientError
import json
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ENDPOINT_NAME = "jumpstart-dft-tf-ic-efficientnet-b3-20250601-080754"
CONTENT_TYPE = "application/x-image"

# Initialize runtime client
try:
    runtime_client = boto3.client("sagemaker-runtime")
except (BotoCoreError, ClientError) as e:
    logger.error("Failed to initialize SageMaker runtime client: %s", str(e))
    raise

def predict_from_image(image_path: str) -> dict:
    """
    Sends an image to the deployed SageMaker endpoint and returns prediction response.
    """
    try:
        # Load image as bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        logger.info("Invoking endpoint: '%s' for image: %s", ENDPOINT_NAME, image_path)

        # Invoke endpoint
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType=CONTENT_TYPE,
            Accept='application/json;verbose',
            Body=image_bytes
        )

        # Process response
        response_body = response.get("Body")
        if response_body is None:
            logger.error("No response body returned from SageMaker endpoint.")
            return {"error": "Empty response"}

        result_json = json.loads(response_body.read())
        probabilities = np.array(result_json["probabilities"])

        top_indices = probabilities.argsort()[-5:][::-1]
        top_probs = probabilities[top_indices]

        with open("models/jetscope-EfficientNetB3-TF-v7/labels_info.json") as f:
            labels = json.load(f)["labels"]

        top_predictions = [
            {
                "class": labels[i],
                "confidence": round(float(prob) * 100, 2),
                "index": int(i)
            }
            for i, prob in zip(top_indices, top_probs)
        ]

        logger.info("Top 5 predictions: %s", top_predictions)
        return {"top_5_predictions": top_predictions}

    except (BotoCoreError, ClientError) as e:
        logger.error("SageMaker client error: %s", str(e))
        return {"error": str(e)}
    except Exception as ex:
        logger.error("Prediction failed: %s", str(ex))
        return {"error": str(ex)}

# Example usage
if __name__ == "__main__":
    test_image = r"D:\Projects\JetScope\data\raw\data\images\1499562.jpg"
    result = predict_from_image(test_image)
    print("Prediction result:", result)
