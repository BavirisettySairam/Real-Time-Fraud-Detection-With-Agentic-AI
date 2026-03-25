from locust import HttpUser, between, task


class FraudApiUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(4)
    def predict(self):
        self.client.post(
            "/api/v1/predict",
            json={
                "TransactionAmt": 120.0,
                "ProductCD": "W",
                "card1": 10001,
                "hour": 13,
                "txn_count_1h": 2,
                "amount_1h": 240.0,
            },
            name="predict_single",
        )

    @task(1)
    def predict_batch(self):
        self.client.post(
            "/api/v1/predict/batch",
            json={
                "transactions": [
                    {"TransactionAmt": 55.0, "ProductCD": "W", "card1": 222},
                    {"TransactionAmt": 6500.0, "ProductCD": "W", "card1": 222, "hour": 3},
                    {"TransactionAmt": 20.0, "ProductCD": "W", "card1": 777},
                ]
            },
            name="predict_batch",
        )
