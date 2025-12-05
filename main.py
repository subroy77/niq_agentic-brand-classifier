import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List

import dspy

from agent.pipeline_agentic import classify_queries


DSPY_MODEL_NAME = os.getenv("DSPY_MODEL_NAME", "phi3")


def configure_lm() -> None:
    """
    Configure DSPy to use Phi-3 via Ollama (local POC).
    Adjust this function only if NIQ wants a different LM backend.
    """
    lm = dspy.Ollama(model=DSPY_MODEL_NAME)
    dspy.settings.configure(lm=lm)


def load_queries_from_file(path: Path) -> List[str]:
    queries: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    return queries


def main() -> None:
    """
    NIQ Agentic Brand Classifier (enhanced POC).

    Behaviour:
    - If ABC_INPUT_FILE is set -> read prompts from that text file
      (one query per non-empty line).
    - Else -> read a single query from stdin.
    - Run agentic pipeline (brand, category, sub-category, validation,
      clarification).
    - Write results to output/brand_results_agentic_*.csv and echo a few
      to console.
    """

    configure_lm()

    input_file_env = os.getenv("ABC_INPUT_FILE", "").strip()
    if input_file_env:
        input_path = Path(input_file_env)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        queries = load_queries_from_file(input_path)
    else:
        print("ABC (agentic) – no ABC_INPUT_FILE set, please type a single query:")
        user_query = input("> ").strip()
        if not user_query:
            print("Empty query. Exiting.")
            return
        queries = [user_query]

    print(f"Classifying {len(queries)} query(ies) with agentic pipeline...")

    results = classify_queries(queries)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"brand_results_agentic_{timestamp}.csv"

    fieldnames = [
        "query",
        "brand",
        "category",
        "category_confidence",
        "sub_category",
        "sub_category_confidence",
        "is_consistent",
        "validation_reason",
        "needs_clarification",
        "clarification_question",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    print(f"Done. Enhanced POC results written to: {csv_path}")

    print("\nSample results:")
    for r in results[:5]:
        print("---------------------------------------------------")
        print(f"Query: {r.query}")
        print(f"  Brand: {r.brand}")
        print(f"  Category: {r.category} (conf={r.category_confidence:.2f})")
        print(
            f"  Sub-category: {r.sub_category} "
            f"(conf={r.sub_category_confidence:.2f})"
        )
        print(f"  Consistent? {r.is_consistent} – {r.validation_reason}")
        if r.needs_clarification:
            print(f"  Clarification question: {r.clarification_question}")


if __name__ == "__main__":
    main()
