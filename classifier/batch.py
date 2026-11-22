"""Batch document classifier with CSV in/out and transparent evaluation.

Run: python classifier/batch.py examples/documents.csv --output predictions.csv
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from document_classifier import classify


def predict_rows(input_path: Path) -> list[dict[str, str | int]]:
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        needed = {"document_id", "text"}
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError(f"Missing columns: {sorted(needed - set(reader.fieldnames or []))}")
        result = []
        seen = set()
        for line, row in enumerate(reader, 2):
            doc_id = row.get("document_id")
            contents = row.get("text")
            if doc_id is None or contents is None or not doc_id.strip():
                raise ValueError(f"Invalid record at line {line}")
            if doc_id in seen:
                raise ValueError(f"Duplicate document ID at line {line}")
            seen.add(doc_id)
            category, scores = classify(contents)
            result.append({"document_id": doc_id, "category": category,
                           "matched_keywords": scores.get(category, 0)})
    return result


def evaluate(input_path: Path) -> dict:
    """Micro accuracy and per-label precision/recall using optional label column."""
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError("Evaluation requires 'text' and 'label' columns")
        actual, predicted = [], []
        for row in reader:
            if not row["label"]:
                raise ValueError("Missing true label")
            actual.append(row["label"])
            predicted.append(classify(row["text"] or "")[0])
    labels = sorted(set(actual) | set(predicted))
    results = {}
    for label in labels:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        results[label] = {
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "support": sum(a == label for a in actual),
        }
    return {"documents": len(actual),
            "accuracy": round(sum(a == p for a, p in zip(actual, predicted)) / len(actual), 4)
            if actual else None, "labels": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, help="Save predictions to CSV")
    parser.add_argument("--evaluate", action="store_true", help="Use true 'label' column to evaluate")
    args = parser.parse_args()
    try:
        if args.evaluate:
            import json
            print(json.dumps(evaluate(args.input_csv), indent=2))
        else:
            results = predict_rows(args.input_csv)
            if args.output:
                with args.output.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["document_id", "category", "matched_keywords"])
                    writer.writeheader()
                    writer.writerows(results)
                print(f"Wrote {len(results)} predictions to {args.output}")
            else:
                for row in results:
                    print(f"{row['document_id']}\t{row['category']}\t{row['matched_keywords']}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
