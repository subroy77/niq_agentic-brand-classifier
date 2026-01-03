import boto3
import json
from .config import AWS_REGION, BEDROCK_MODEL_ID


def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def invoke_bedrock(prompt: str, model_id: str = BEDROCK_MODEL_ID):
    client = get_bedrock_client()

    body = json.dumps({
        "inputText": prompt
    })

    response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
    )

    result = json.loads(response["body"].read())
    return result.get("outputText", "")
