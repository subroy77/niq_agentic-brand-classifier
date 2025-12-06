import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List

import dspy

from agent.pipeline_agentic import classify_queries, classify_single_query


# -----------------------------------------------------------------------------
# 1. Configure DSPy + Phi-3 via Ollama (LOCAL POC)
# -----------------------------------------------------------------------------

# These env vars keep existing behaviour if NIQ already uses them
DSPY_MODEL_PROVIDER = os.getenv("DSPY_MODEL_PROVIDER", "ollama")  # for future use
DSPY_MODEL_NAME = os.getenv("DSPY_MODEL_NAME", "phi3")

# For now we assume Ollama + Phi-3, as in the original POC
lm = dspy.Ollama(model=DSPY_MODEL_NAME)
dspy.settings.configure(lm=lm)


# -----------------------------------------------------------------------------
# 2. Helper: load queries
# -----------------------------------------------------------------------------

def load_queries_from_file(path: Path) -> List[str]:
    """
    Simple loader: one user prompt per non-empty line.
    """
    queries: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    return queries


# -----------------------------------------------------------------------------
# 3. Main: run classification and write enhanced CSV to output/
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Entry point for the NIQ Agentic Brand Classifier (enhanced POC).
    - Reads prompts from a text file if ABC_INPUT_FILE is set (one per line).
    - Otherwise, reads a single query from stdin.
    - Runs the agentic pipeline (brand, category, sub-category, validation,
      clarification).
    - Writes results to output/brand_results_agentic_*.csv
    """

    input_file_env = os.getenv("ABC_INPUT_FILE", "").strip()
    if input_file_env:
        input_path = Path(input_file_env)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        queries = load_queries_from_file(input_path)
    else:
        # Single interactive query mode
        print("ABC (agentic) – no ABC_INPUT_FILE set, please type a single query:")
        user_query = input("> ").strip()
        if not user_query:
            print("Empty query. Exiting.")
            return
        queries = [user_query]

    print(f"Classifying {len(queries)} query(ies) with agentic pipeline...")

    # Run pipeline
    results = classify_queries(queries)

    # Ensure output directory exists
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

    # Also print results to console for immediate inspection
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
