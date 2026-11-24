import unittest

from src.predict import predict


class TestPredictionPipeline(unittest.TestCase):

    def test_prediction_returns_dictionary(self):
        result = predict(
            "Apple announced new artificial intelligence features for its devices."
        )
        self.assertIsInstance(result, dict)

    def test_prediction_contains_expected_fields(self):
        result = predict(
            "The football team won the championship after a dramatic final."
        )

        self.assertIn("category", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)

    def test_label_is_valid(self):
        result = predict(
            "Global leaders met to discuss international relations and security."
        )

        self.assertIn(
            result["category"],
            {"World", "Sports", "Business", "Sci/Tech"},
        )

    def test_confidence_is_valid_probability(self):
        result = predict(
            "Stocks rose after the company reported strong quarterly earnings."
        )

        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_probabilities_cover_all_classes(self):
        result = predict(
            "Researchers unveiled a new processor for artificial intelligence."
        )

        probabilities = result["probabilities"]

        self.assertEqual(
            set(probabilities.keys()),
            {"World", "Sports", "Business", "Sci/Tech"},
        )

    def test_probabilities_sum_to_one(self):
        result = predict(
            "The central bank announced changes to interest rates."
        )

        total = sum(result["probabilities"].values())

        self.assertAlmostEqual(total, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()