import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    os.makedirs("artifacts", exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training MNIST MLP...")
    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss / len(trainloader):.4f}")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

    np.save("artifacts/fc1.weight.npy", model.fc1.weight.detach().numpy())
    np.save("artifacts/fc1.bias.npy", model.fc1.bias.detach().numpy())
    np.save("artifacts/fc2.weight.npy", model.fc2.weight.detach().numpy())
    np.save("artifacts/fc2.bias.npy", model.fc2.bias.detach().numpy())

    sample_img, sample_label = next(iter(testloader))
    np.save("artifacts/sample_image.npy", sample_img.detach().numpy())
    
    outputs = model(sample_img)
    _, predicted = torch.max(outputs.data, 1)
    np.savetxt("artifacts/expected_pred.txt", [predicted.item()], fmt="%d")
    np.save("artifacts/expected_logits.npy", outputs.detach().numpy())

    print(f"Exported artifacts. Expected prediction: {predicted.item()}")

if __name__ == "__main__":
    main()
