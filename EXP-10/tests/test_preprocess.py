import unittest

import pandas as pd

from src.preprocess import clean_pairs, clean_text, split_pairs


class TestPreprocessing(unittest.TestCase):

    def test_clean_text_normalizes_whitespace(self):
        self.assertEqual(clean_text("  hello   world  "), "hello world")

    def test_clean_text_handles_non_string(self):
        self.assertEqual(clean_text(None), "")

    def test_clean_pairs_removes_empty_and_duplicates(self):
        df = pd.DataFrame({
            "src": ["hello", "hello", "", "world"],
            "tgt": ["নমস্কাৰ", "নমস্কাৰ", "text", "বিশ্ব"],
        })

        cleaned, stats = clean_pairs(df)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(stats["after_dedup"], 2)

    def test_split_is_deterministic(self):
        df = pd.DataFrame({
            "src": [f"src-{i}" for i in range(100)],
            "tgt": [f"tgt-{i}" for i in range(100)],
        })

        a = split_pairs(df, seed=42)
        b = split_pairs(df, seed=42)

        for left, right in zip(a, b):
            pd.testing.assert_frame_equal(
                left.reset_index(drop=True),
                right.reset_index(drop=True),
            )

    def test_split_sizes(self):
        df = pd.DataFrame({
            "src": [str(i) for i in range(100)],
            "tgt": [str(i) for i in range(100)],
        })

        train, val, test = split_pairs(df, seed=42)

        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(test), 10)


if __name__ == "__main__":
    unittest.main()
