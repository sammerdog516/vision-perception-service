import requests
from torchvision import datasets, transforms

ds = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

img, label = ds[0]
img2d = img.squeeze(0).tolist()

resp = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"image": img2d}
)

print("True", label)
print("Status:", resp.status_code)
print("Raw response text:")
print(resp.text)