import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestRepository(unittest.TestCase):

    def test_notebooks_have_python_kernel_metadata(self):
        notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
        self.assertTrue(notebooks)

        for path in notebooks:
            with self.subTest(notebook=path.name):
                nb = json.loads(path.read_text(encoding="utf-8"))

                self.assertEqual(
                    nb["metadata"]["kernelspec"]["name"],
                    "python3",
                )
                self.assertEqual(
                    nb["metadata"]["language_info"]["name"],
                    "python",
                )

    def test_notebooks_are_valid_nbformat_4(self):
        for path in (ROOT / "notebooks").glob("*.ipynb"):
            with self.subTest(notebook=path.name):
                nb = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(nb["nbformat"], 4)
                self.assertIsInstance(nb["cells"], list)

    def test_large_training_artifacts_are_not_tracked(self):
        forbidden_dirs = {"data", "nllb-en-as"}
        forbidden_suffixes = {".safetensors", ".bin", ".pt"}

        for path in ROOT.rglob("*"):
            if ".git" in path.parts or not path.is_file():
                continue

            relative = path.relative_to(ROOT)

            self.assertFalse(
                any(part in forbidden_dirs for part in relative.parts),
                f"Forbidden training artifact found: {relative}",
            )

            self.assertNotIn(
                path.suffix,
                forbidden_suffixes,
                f"Forbidden model artifact found: {relative}",
            )


if __name__ == "__main__":
    unittest.main()
