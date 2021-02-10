import requests

def retrieve_products(customer):
    url = f"https://product-recommendation-d6709-default-rtdb.europe-west1.firebasedatabase.app/{customer}.json"
    resp = requests.get(url)
    json_data = resp.json()
    return json_data['products']