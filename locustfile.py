from locust import HttpUser, task, between


class CreditRiskUser(HttpUser):

    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json={
                "text": "This is a test credit application"
            }
        )
