# niq_agentic-brand-classifier

Enhanced NIQ Agentic Brand Classifier POC using DSPy + Phi-3 (via Ollama).

This repo demonstrates:
- Brand + category classification
- Sub-category classification (no fixed taxonomy)
- Confidence scores
- Validation agent (consistency check)
- Clarification agent (asks follow-up question when confidence is low or labels look inconsistent)

## Prerequisites

- Python 3.10+ (tested with 3.11)
- [Ollama](https://ollama.ai/) installed locally
- Phi-3 model pulled in Ollama, e.g.:

```bash
ollama pull phi3
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Running

### Single interactive query

```bash
python main.py
```

You will be prompted to type a single query.

### Batch from file

Create a text file (e.g. `sample_prompts.txt`) with one query per line,
then run:

```bash
export ABC_INPUT_FILE=sample_prompts.txt
python main.py
```

## Output

Results are written to:

```text
output/brand_results_agentic_YYYYMMDD_HHMMSS.csv
```

Each row contains:

- query
- brand
- category
- category_confidence
- sub_category
- sub_category_confidence
- is_consistent
- validation_reason
- needs_clarification
- clarification_question

This repo is self-contained and does not rely on any previous CSV outputs
or taxonomy files.
