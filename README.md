# Explainable Document Classification

A transparent, rule-based document categorizer implemented with Python's standard library.

## Run

```bash
python examples/document_classifier.py
```

The example classifies text into finance, engineering, or support by counting matched keyword sets. It returns both the predicted label and category scores.

This is a deterministic baseline, not an LLM or trained machine-learning classifier. Useful extensions include a labeled evaluation set, precision/recall reports, and ML-based classification.
