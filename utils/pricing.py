import requests

REGIONS = ["us-east-at-1", "ap-south-mum-1", "ap-south-noi-1"]
BASE_URL = "https://customer.acecloudhosting.com/api/v1/pricing?is_gpu=true&resource=instances&region="

def fetch_gpu_data_from_regions():
    combined_data = []
    for region in REGIONS:
        url = BASE_URL + region
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()
            if not json_data["error"]:
                combined_data.extend(json_data["data"])
    return combined_data
