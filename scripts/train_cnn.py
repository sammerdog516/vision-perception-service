import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from app.models.cnn import CNN

transform = transforms.Compose([
    transforms.ToTensor(),
    # (H, W) -> (1, H, W), values in [0,1]
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
def add_gaussian_noise(x, std=0.3):
    noise = torch.rand_like(x) * std
    return x + noise

def shift_right(x, pixels=2):
    return torch.roll(x, shifts=pixels, dims=3) # dim=3 is width

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()         # expects logits
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


epochs = 5 

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            # Add noise and translation ONLY at test time
            x_shifted = shift_right(x, pixels=2)
            x_noisy = add_gaussian_noise(x_shifted, std=0.3)

            logits = model(x_noisy)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    
    print(f"Noisy + Translation Test accuracy: {accuracy*100:.2f}%")
torch.save(model.state_dict(), "weights/cnn_mnist.pt")