import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Preprocess data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Define CNN architecture
class CNN(nn.Module):
    def __init__(self, use_bn_dropout=False):
        super(CNN, self).__init__()
        self.use_bn_dropout = use_bn_dropout
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if self.use_bn_dropout:
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        else:
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        if self.use_bn_dropout:
            x = torch.relu(self.dropout(self.fc1(x)))
        else:
            x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define model train function
def train_model(model, trainloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

# Define model test function
def test_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy

# Initialize models, criterion, and optimizers
model_without_bn_dropout = CNN(use_bn_dropout=False)
optimizer_without_bn_dropout = optim.Adam(model_without_bn_dropout.parameters(), lr=0.001)

model_with_bn_dropout = CNN(use_bn_dropout=True)
optimizer_with_bn_dropout = optim.Adam(model_with_bn_dropout.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

# Train and test model without BatchNorm and Dropout
print("Training CNN without BatchNorm and Dropout...")
train_model(model_without_bn_dropout, trainloader, criterion, optimizer_without_bn_dropout, epochs=5)
print("Testing CNN without BatchNorm and Dropout...")
test_model(model_without_bn_dropout, testloader)

# Train and test model with BatchNorm and Dropout
print("Training CNN with BatchNorm and Dropout...")
train_model(model_with_bn_dropout, trainloader, criterion, optimizer_with_bn_dropout, epochs=5)
print("Testing CNN with BatchNorm and Dropout...")
test_model(model_with_bn_dropout, testloader)
