from fastapi import FastAPI
from .service import classify_single_record

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/classify")
def classify(payload: dict):
    """
    Expected input (one CSV row as JSON):

    {
      "record": {
        "product_id": "123",
        "description": "Organic almond butter 500g"
      }
    }
    """

    record = payload.get("record")

    if not record:
        return {"error": "Missing 'record' field in request body"}

    try:
            result = classify_single_record(record)
            return result

    except Exception as e:
            return {"error": str(e)}
