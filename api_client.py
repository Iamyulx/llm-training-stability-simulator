import requests

class APIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.endpoints = [
            "https://router.huggingface.co/v1/chat/completions",
            "https://backup-endpoint.com/v1/chat/completions"
        ]

    def request(self, payload):
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for endpoint in self.endpoints:
            try:
                response = requests.post(endpoint, json=payload, headers=headers)

                if response.status_code == 410:
                    print(f"⚠️ Endpoint deprecated: {endpoint}")
                    continue

                response.raise_for_status()
                return response.json()

            except Exception as e:
                print(f"Error with {endpoint}: {e}")

        raise RuntimeError("All endpoints failed")
