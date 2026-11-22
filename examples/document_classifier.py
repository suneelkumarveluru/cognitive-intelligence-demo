"""Rule-based document classifier for an interpretable AI demo."""
import re

KEYWORDS = {
    "finance": {"invoice", "revenue", "payment", "ledger"},
    "engineering": {"pipeline", "schema", "database", "deployment"},
    "support": {"ticket", "incident", "customer", "resolution"},
}


def classify(text: str) -> tuple[str, dict[str, int]]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    scores = {category: len(words & keywords)
              for category, keywords in KEYWORDS.items()}
    if not scores or max(scores.values()) == 0:
        return "unknown", scores
    best = max(sorted(scores), key=scores.get)
    return best, scores


if __name__ == "__main__":
    print(classify("The schema deployment failed during the pipeline run"))
