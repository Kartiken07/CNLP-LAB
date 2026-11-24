import unittest

from app.app import app


class TestFlaskApplication(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIsInstance(data, dict)

    def test_model_endpoint(self):
        response = self.client.get("/api/model")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIsInstance(data, dict)

    def test_predict_endpoint(self):
        response = self.client.post(
            "/api/predict",
            json={
                "text": (
                    "The football team won the championship "
                    "after a dramatic final."
                )
            },
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()

        self.assertIn("category", data)
        self.assertIn("confidence", data)
        self.assertIn("probabilities", data)

        self.assertIn(
            data["category"],
            {"World", "Sports", "Business", "Sci/Tech"},
        )

    def test_predict_rejects_missing_text(self):
        response = self.client.post(
            "/api/predict",
            json={},
        )

        self.assertGreaterEqual(response.status_code, 400)
        self.assertLess(response.status_code, 500)

    def test_predict_rejects_empty_text(self):
        response = self.client.post(
            "/api/predict",
            json={"text": ""},
        )

        self.assertGreaterEqual(response.status_code, 400)
        self.assertLess(response.status_code, 500)
    
    def test_frontend_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)


    def test_predict_rejects_excessively_long_text(self):
        response = self.client.post(
            "/api/predict",
            json={"text": "a" * 20001},
        )

        self.assertEqual(response.status_code, 413)


    def test_model_endpoint_contains_metadata(self):
        response = self.client.get("/api/model")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(data, dict)
        self.assertTrue(len(data) > 0)


if __name__ == "__main__":
    unittest.main()