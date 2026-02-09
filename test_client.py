import requests

url = "http://127.0.0.1:8080/predict"

with open("test_face.jpg", "rb") as f:
    image_bytes = f.read()

response = requests.post(url, data=image_bytes)

print(response.status_code)
print(response.json())
