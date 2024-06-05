from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from cnn import SimpleCNN

IMG_HEIGHT = 50
IMG_WIDTH = 50
TEST_DATA_PATH = 'Test Data'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT,IMG_WIDTH)),
    transforms.ToTensor(),
])

# Create dataset instance for testing
test_dataset = ImageFolder(TEST_DATA_PATH, transform=transform)

# Create dataloader instance for testing
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Creates an instance of the model
num_classes = 2 # 2 because 2 instances broken and not broken
model = SimpleCNN(num_classes)

# Load the model
model.load_state_dict(torch.load('trainedModel.pth'))

# Switch the model to evaluation mode
model.eval()

# Loop over the test data
for i, (inputs, _) in enumerate(test_loader):
    outputs = model(inputs)

    _, predicted = torch.max(outputs.data, 1)

    for j, prediction in enumerate(predicted):
        index = i * test_loader.batch_size + j
        filename = test_dataset.samples[index][0]

        # Print the filename and the prediction
        print(f"Image: {filename}, Prediction: {'Broken' if prediction.item() == 1 else 'Not Broken'}")