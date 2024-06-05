import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import optim, nn
from cnn import SimpleCNN

TRAIN_DATA_PATH = 'Train Data'

IMG_HEIGHT = 50
IMG_WIDTH = 50

# Define transformations
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT,IMG_WIDTH)),
    transforms.ToTensor(),
])

# Create dataset instances
train_dataset = ImageFolder(TRAIN_DATA_PATH, transform=transform)

# Create dataloader instances
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Create an instance of the model
num_classes = 2
model = SimpleCNN(num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs to train for
num_epochs = 10

# Loop over the dataset multiple times
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1} loss: {running_loss/(i+1)}")

print('Finished Training')

torch.save(model.state_dict(), 'model.pth')
print('Model saved to model.pth')
