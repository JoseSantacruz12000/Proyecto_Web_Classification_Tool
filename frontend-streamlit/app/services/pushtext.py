import requests
import os
import json

class Web_ruted:
    def pushpost(url: str):

        payload = json.dumps({"URL": url})
        headers = {"Content-Type": "application/json"}
        url_api = f"{os.environ['API_ENDPOINT']}/web_clasification/predictions"
        response = requests.request("POST", url_api, headers=headers, data=payload)
        return response, response.json()