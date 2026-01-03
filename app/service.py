from .bedrock_client import invoke_bedrock
from .config import CSV_TEXT_COLUMN


def extract_text_from_csv_row(record: dict) -> str:
    """
    record = single CSV row converted to dict

    Example:
        {
          "product_id": "123",
          "brand": "Nike",
          "description": "Ultra-light running shoes"
        }
    """

    if CSV_TEXT_COLUMN not in record:
        raise ValueError(
            f"CSV column '{CSV_TEXT_COLUMN}' was not found in payload."
        )

    return record[CSV_TEXT_COLUMN]


def run_brand_classification(text: str) -> str:
    """
    Call your internal logic here.
    For PoC we route text → Bedrock prompt.
    """

    prompt = f"""
    Classify the brand and category from this product description:

    {text}

    Respond in JSON with:
    - brand
    - category
    - confidence
    """

    return invoke_bedrock(prompt)


def classify_single_record(record: dict):
    text = extract_text_from_csv_row(record)
    result = run_brand_classification(text)

    return {
        "input_text": text,
        "classification": result
    }
