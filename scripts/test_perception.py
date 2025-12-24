import torch
import numpy as np
from torchvision import datasets, transforms

from app.perception.cnn_perception import load_model, predict

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("weights/cnn_mnist.pt", device)

    # Load one MNIST sample as a stand-in for a sensor frame
    ds = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    img, label = ds[0]              # img: (1, 28, 28) tensor in [0, 1]
    img2d = img.squeeze(0).numpy()  # (28, 28)

    digit, conf = predict(model, img2d, device)
    print("True:", label, "| Pred:", digit, "| Conf:", conf)

    # blank frame sanity check
    blank = np.zeros((28, 28), dtype=np.float32)
    digit2, conf2 = predict(model, blank, device)
    print("Blank -> Pred:", digit2, "| Conf:", conf2)

if __name__ == "__main__":
    main()