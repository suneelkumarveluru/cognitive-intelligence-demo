# Explainable batch document classification

Python 3.10+; no dependencies beyond the standard library.

## Try it

```bash
python classifier/batch.py examples/documents.csv
python classifier/batch.py examples/documents.csv --output predictions.csv
python classifier/batch.py examples/documents.csv --evaluate
python -m unittest discover -s tests -v
```

Input requires `document_id,text`; optional `label` is the ground-truth category used by `--evaluate`. Output includes predicted categories and the number of distinct matching keywords.

Evaluation prints accuracy, per-class precision, recall, and label support. These metrics are based on the labeled sample and do not demonstrate generalization.

**Limits:** The classifier is deterministic and keyword-based, not trained ML or an LLM. Labels currently cover finance, engineering, support and unknown. It uses set membership rather than frequency or semantic understanding. Do not use this baseline to automate consequential decisions or process confidential documents without a privacy review.
