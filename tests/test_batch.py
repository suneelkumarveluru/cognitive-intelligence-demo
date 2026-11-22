import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classifier"))
from batch import predict_rows, evaluate


class BatchTests(unittest.TestCase):
    def test_predict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "docs.csv"
            path.write_text("document_id,text\n1,invoice payment\n2,pipeline schema\n", encoding="utf-8")
            results = predict_rows(path)
            self.assertEqual([x["category"] for x in results], ["finance", "engineering"])

    def test_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "docs.csv"
            path.write_text("text,label\ninvoice payment,finance\npipeline schema,engineering\n", encoding="utf-8")
            report = evaluate(path)
            self.assertEqual(report["accuracy"], 1.0)
            self.assertEqual(report["labels"]["finance"]["support"], 1)

    def test_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "docs.csv"
            path.write_text("document_id,text\n1,test\n1,test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                predict_rows(path)

    def test_no_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "docs.csv"
            path.write_text("text,label\n", encoding="utf-8")
            self.assertIsNone(evaluate(path)["accuracy"])


if __name__ == "__main__":
    unittest.main()
