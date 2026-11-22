import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from document_classifier import classify


class ClassifierTests(unittest.TestCase):
    def test_engineering_category(self):
        label, scores = classify("schema pipeline")
        self.assertEqual(label, "engineering")
        self.assertEqual(scores["engineering"], 2)

    def test_unknown_document(self):
        self.assertEqual(classify("rainy weather forecast")[0], "unknown")

    def test_case_independent(self):
        self.assertEqual(classify("INVOICE payment")[0], "finance")


if __name__ == "__main__":
    unittest.main()
