import os

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# CSV → which column should be classified
CSV_TEXT_COLUMN = os.getenv("CSV_TEXT_COLUMN", "description")

# Default Bedrock model
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-sonnet-20240229-v1:0"
)
