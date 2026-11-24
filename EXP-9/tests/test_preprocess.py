import unittest

from src.preprocess import clean_text


class TestPreprocessing(unittest.TestCase):

    def test_returns_string(self):
        result = clean_text("Business markets are growing rapidly.")
        self.assertIsInstance(result, str)

    def test_lowercases_text(self):
        result = clean_text("WORLD ECONOMY")
        self.assertEqual(result, result.lower())

    def test_removes_url(self):
        result = clean_text(
            "Read this article https://example.com about technology"
        )
        self.assertNotIn("http", result)
        self.assertNotIn("example.com", result)

    def test_removes_html(self):
        result = clean_text("<p>Sports championship result</p>")
        self.assertNotIn("<p>", result)
        self.assertNotIn("</p>", result)

    def test_removes_special_characters(self):
        result = clean_text("Markets!!! stocks??? profits $$$")
        self.assertNotIn("!", result)
        self.assertNotIn("?", result)
        self.assertNotIn("$", result)

    def test_handles_empty_text(self):
        result = clean_text("")
        self.assertIsInstance(result, str)

    def test_is_deterministic(self):
        text = "Technology companies announced new AI products."
        self.assertEqual(
            clean_text(text),
            clean_text(text),
        )


if __name__ == "__main__":
    unittest.main()
