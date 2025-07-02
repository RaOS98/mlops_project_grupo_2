from sagemaker import image_uris

IMAGE_URI = image_uris.retrieve(
    framework="xgboost",
    region="us-east-2",
    version="1.7-1"
)

INSTANCE_TYPE = "ml.m5.large"